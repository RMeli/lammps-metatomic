/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS Development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors: Guillaume Fraux <guillaume.fraux@epfl.ch>
------------------------------------------------------------------------- */
#include "metatomic_types.h"

#include "atom.h"
#include "citeme.h"
#include "comm.h"
#include "error.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>

#include <metatomic/metadata.hpp>
#include <metatomic/plugin.hpp>

using namespace LAMMPS_NS;

// Name of a metatomic device, matching the strings used by metatomic
static std::string device_name(metatomic::ModelCapabilities::Device device) {
    switch (device) {
        case metatomic::ModelCapabilities::Device::CPU: return "cpu";
        case metatomic::ModelCapabilities::Device::CUDA: return "cuda";
        case metatomic::ModelCapabilities::Device::ROCm: return "rocm";
        case metatomic::ModelCapabilities::Device::Metal: return "metal";
        default: return "cpu";
    }
}

CommonMetatomicData::CommonMetatomicData(std::string length_unit):
    capabilities(
        metatomic::ModelCapabilities::builder()
            .atomic_types({})
            .interaction_range(-1)
            .length_unit(length_unit)
            .supported_devices({})
            .dtype(metatomic::ModelCapabilities::DType::Float32)
            .outputs({})
            .build()
    ) {}

PairMetatomicData::PairMetatomicData(std::string length_unit): CommonMetatomicData(std::move(length_unit)){
    this->energy_output = metatomic::Quantity::builder()
        .name("energy")
        .unit("eV")
        .sample_kind(metatomic::SampleKind::System)
        .build();

    this->uncertainty_output = metatomic::Quantity::builder()
        .name("uncertainty")
        .unit("eV")
        .sample_kind(metatomic::SampleKind::System)
        .build();

    this->nc_forces_output = metatomic::Quantity::builder()
        .name("nc_forces")
        .unit("eV/angstrom")
        .sample_kind(metatomic::SampleKind::System)
        .build();

    this->nc_stress_output = metatomic::Quantity::builder()
        .name("nc_stress")
        .unit("eV/angstrom^3")
        .sample_kind(metatomic::SampleKind::System)
        .build();

}

void CommonMetatomicData::load_model(
   LAMMPS* lmp,
   const char* path,
   const char* extensions_directory
) {
   // TODO: seach for the model & extensions inside `$LAMMPS_POTENTIALS`?

   this->model_path = path;
   if (this->model != nullptr) {
       lmp->error->one(FLERR, "torch model is already loaded");
   }

   std::optional<std::string> extensions = std::nullopt;
   if (extensions_directory != nullptr) {
       extensions = std::string(extensions_directory);
   }

   try {
       this->model = std::make_unique<metatomic::ExternalModel>(
           metatomic::load_model(this->model_path, extensions)
       );
   } catch (const metatomic::Error& e) {
       lmp->error->one(FLERR, "failed to load metatomic model at '{}': {}", path, e.what());
   }

   this->capabilities = this->model->capabilities();

   if (lmp->comm->me == 0) {
       auto metadata = this->model->metadata();

       if (lmp->screen) {
           fprintf(lmp->screen, "\n%s\n", metatomic::format_metadata(metadata.to_json()).c_str());
       }
       if (lmp->logfile) {
           fprintf(lmp->logfile,"\n%s\n", metatomic::format_metadata(metadata.to_json()).c_str());
       }

       // add the model references to LAMMPS citation handling mechanism
       if (lmp->citeme) {
          for (const auto& it: metadata.references()) {
             for (const auto& ref: it.second) {
                lmp->citeme->add(ref + "\n");
             }
          }
       }
   }
}

void CommonMetatomicData::check_requested_device(LAMMPS* lmp, const char* requested, const char* cmd_name) const {
    // The data LAMMPS builds for the model is on the CPU.
    // Moving it to/from the GPU is up to the model itself.
    if (requested == nullptr) {
        return;
    }

    auto wanted = std::string(requested);
    auto supported = this->capabilities.supported_devices();
    auto it = std::find_if(supported.begin(), supported.end(), [&](const auto& d) {
        return device_name(d) == wanted;
    });

    // device not supported
    if (it == supported.end()) {
        std::string supported_names;
        for (const auto& d: supported) {
            if (!supported_names.empty()) {
                supported_names += ", ";
            }
            supported_names += device_name(d);
        }
        lmp->error->one(FLERR,
            "{}: the model does not support the requested device '{}'; supported devices are: {}",
            cmd_name, wanted, supported_names
        );
    }
}

DLDataType CommonMetatomicData::model_dtype(LAMMPS* lmp) const {
    const auto dtype = this->capabilities.dtype();
    if (dtype == metatomic::ModelCapabilities::DType::Float64) {
        return DLDataType{kDLFloat, 64, 1};
    } else if (dtype == metatomic::ModelCapabilities::DType::Float32) {
        return DLDataType{kDLFloat, 32, 1};
    } else {
        lmp->error->all(FLERR, "the model requested an unsupported dtype");
    }
    return DLDataType{kDLFloat, 64, 1};  // unreachable, error->all does not return
}

void CommonMetatomicData::resolve_max_cutoff(LAMMPS* lmp) {
    // get the model's interaction range in the model length unit
    auto range = this->capabilities.interaction_range();
    // TODO RM: convert the interaction range to LAMMPS units
    if (range < 0) {
        lmp->error->all(FLERR, "interaction_range is negative for this model");
    } else if (!std::isfinite(range)) {
        if (lmp->comm->nprocs > 1) {
            lmp->error->all(FLERR,
                "interaction_range is infinite for this model, "
                "using multiple MPI domains is not supported"
            );
        }

        // determine the maximal cutoff in the NL
        for (const auto& options: this->model->requested_pair_lists()) {
            auto cutoff = options.cutoff();
            this->max_cutoff = std::max(this->max_cutoff, cutoff);
        }
    } else {
        this->max_cutoff = range;
    }
}

// void CommonMetatomicData::set_selected_atoms(Atom* atom, int groupbit) {
//     int nlocal = atom->nlocal;
//     int *mask = atom->mask;
//
//     int64_t n_selected = 0;
//     for (int i = 0; i < nlocal; i++) {
//         if (mask[i] & groupbit) {
//             n_selected++;
//         }
//     }
//     this->selected_atoms_values_cpu.resize_({n_selected, 2});
//     auto accessor = this->selected_atoms_values_cpu.accessor<int32_t, 2>();
//     int64_t idx = 0;
//     for (int i = 0; i < nlocal; i++) {
//         if (mask[i] & groupbit) {
//             accessor[idx][0] = 0;
//             accessor[idx][1] = i;
//             idx++;
//         }
//     }
//     this->selected_atoms_values.resize_({n_selected, 2});
//     this->selected_atoms_values.copy_(this->selected_atoms_values_cpu);
//
//     auto selected_atoms = torch::make_intrusive<metatensor_torch::LabelsHolder>(
//         std::vector<std::string>{"system", "atom"}, this->selected_atoms_values
//     );
//     this->evaluation_options->set_selected_atoms(selected_atoms);
// }

std::map<std::string, metatomic::Quantity> CommonMetatomicData::collect_requested_inputs() const {
    std::map<std::string, metatomic::Quantity> input_holders;
    for (const auto& quantity: this->model->requested_inputs()) {
        input_holders.emplace(quantity.name(), quantity);
    }
    return input_holders;
}
