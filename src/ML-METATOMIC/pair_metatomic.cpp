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
                         Rocco Meli <rocco.meli@cscs.ch>
------------------------------------------------------------------------- */
#include "pair_metatomic.h"
#include "metatomic_types.h"

#include "atom.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neighbor.h"
#include "update.h"
#include "comm.h"
#include "domain.h"

#include "neigh_list.h"

#include <map>
#include <memory>
#include <optional>
#include <string>

// TODO RM: fix nlohmann
#include "json.h"
namespace nlohmann = ::nlohmann_lmp;

#include <metatomic/model.hpp>
#include <metatensor.hpp>

#include "metatomic_system.h"
#include "metatomic_timer.h"
#include "metatomic_quantities.h"

using namespace LAMMPS_NS;

static std::optional<std::string> normalize_variant(const char* variant_string) {
    if (variant_string == nullptr) {
        return std::nullopt;
    } else if (strcmp(variant_string, "off") == 0) {
        return std::nullopt;
    } else {
        return std::string(variant_string);
    }
}

static double compute_volume(Domain* domain) {
    // from Thermo::compute_vol
    if (domain->dimension == 3) {
        return domain->xprd * domain->yprd * domain->zprd;
    } else {
        return domain->xprd * domain->yprd;
    }
}

// Resolve the output key matching `name` or `name/variant` among the
// model's outputs.
// TODO RM: Move to metatomic?
static std::string pick_output(
    const std::string& name,
    const std::vector<metatomic::Quantity>& outputs,
    const std::optional<std::string>& variant
) {
    auto full_name = name;
    if (variant.has_value() && !variant->empty()) {
        full_name += "/" + variant.value();
    }

    for (const auto& output: outputs) {
        if (output.name() == full_name) {
            return output.name();
        }
    }

    // no match: build the list of available variants for the error message
    std::string available;
    for (const auto& output: outputs) {
        if (output.name().rfind(name + "/", 0) == 0) {
            if (!available.empty()) {
                available += ", ";
            }
            available += output.name();
        }
    }

    throw std::runtime_error(
        "no output named '" + full_name + "' in this model"
        + (available.empty() ? "" : " (available: " + available + ")")
    );
}

PairMetatomic::PairMetatomic(LAMMPS *lmp):
    Pair(lmp),
    type_mapping(nullptr),
    system_adaptor(nullptr),
    scale(1.0)
{
    if (strcmp(update->unit_style, "lj") == 0) {
        error->one(FLERR, "unsupported units '{}' for pair metatomic ", update->unit_style);
    }
    this->length_unit = metatomic_unit_map.at("position").at(update->unit_style);
    this->energy_unit = metatomic_unit_map.at("energy").at(update->unit_style);

    // we might not be running a pure pair potential,
    // so we can not compute virial as fdotr
    this->no_virial_fdotr_compute = 1;

    this->mta_data = new PairMetatomicData(this->length_unit);
    // use a default uncertainty threshold of 100 meV/atom
    this->mta_data->uncertainty_threshold = 0.1 * metatomic::unit_conversion_factor("eV", energy_unit);

    // settings for metatomic pair style
    this->single_enable = 0;
    this->restartinfo = 0;
    this->one_coeff = 1;
    this->manybody_flag = 1;
}

PairMetatomic::~PairMetatomic() {
    delete this->mta_data;

    if (allocated) {
        memory->destroy(setflag);
        memory->destroy(cutsq);
        memory->destroy(type_mapping);
    }
}

// called when finding `pair_style metatomic` in the input
void PairMetatomic::settings(int argc, char ** argv) {
    if (argc == 0) {
        error->one(FLERR, "expected at least 1 argument to pair_style metatomic, got {}", argc);
    }

    const char* model_path = argv[0];
    const char* extensions_directory = nullptr;
    const char* requested_device = nullptr;
    bool do_uncertainty = true;
    const char* variant = nullptr;
    const char* variant_energy = nullptr;
    const char* variant_energy_uq = nullptr;
    const char* variant_nc_forces = nullptr;
    const char* variant_nc_stress = nullptr;
    for (int i=1; i<argc; i++) {
        if (strcmp(argv[i], "check_consistency") == 0) {
            if (i == argc - 1) {
                error->one(FLERR, "expected <on/off> after 'check_consistency' in pair_style metatomic, got nothing");
            } else if (strcmp(argv[i + 1], "on") == 0) {
                mta_data->check_consistency = true;
            } else if (strcmp(argv[i + 1], "off") == 0) {
                mta_data->check_consistency = false;
            } else {
                error->one(FLERR, "expected <on/off> after 'check_consistency' in pair_style metatomic, got '{}'", argv[i + 1]);
            }

            i += 1;
        } else if (strcmp(argv[i], "non_conservative") == 0) {
            if (i == argc - 1) {
                error->one(FLERR, "expected <on/off/forces/stress> after 'non_conservative' in pair_style metatomic, got nothing");
            } else if (strcmp(argv[i + 1], "on") == 0) {
                mta_data->non_conservative = PairMetatomicData::NonConservativeMode::ON;
            } else if (strcmp(argv[i + 1], "off") == 0) {
                mta_data->non_conservative = PairMetatomicData::NonConservativeMode::OFF;
            } else if (strcmp(argv[i + 1], "forces") == 0) {
                mta_data->non_conservative = PairMetatomicData::NonConservativeMode::FORCES;
            } else if (strcmp(argv[i + 1], "stress") == 0) {
                mta_data->non_conservative = PairMetatomicData::NonConservativeMode::STRESS;
            } else {
                error->one(FLERR, "expected <on/off/forces/stress> after 'non_conservative' in pair_style metatomic, got '{}'", argv[i + 1]);
            }

            i += 1;
        } else if (strcmp(argv[i], "extensions") == 0) {
            if (i == argc - 1) {
                error->one(FLERR, "expected <path> after 'extensions' in pair_style metatomic, got nothing");
            }
            extensions_directory = argv[i + 1];
            i += 1;
        } else if (strcmp(argv[i], "device") == 0) {
            if (i == argc - 1) {
                error->one(FLERR, "expected string after 'device' in pair_style metatomic, got nothing");
            }
            requested_device = argv[i + 1];
            i += 1;
        } else if (strcmp(argv[i], "scale") == 0) {
            if (i == argc - 1) {
                error->one(FLERR, "expected a number after 'scale' in pair_style metatomic, got nothing");
            }
            this->scale = utils::numeric(FLERR, argv[i + 1], false, lmp);
            i += 1;
        } else if (strcmp(argv[i], "uncertainty_threshold") == 0) {
            if (i == argc - 1) {
                error->one(FLERR, "expected a number or off after 'uncertainty_threshold' in pair_style metatomic, got nothing");
            } else if (strcmp(argv[i + 1], "off") == 0) {
                do_uncertainty = false;
            } else {
                mta_data->uncertainty_threshold = utils::numeric(FLERR, argv[i + 1], false, lmp);
            }

            if (mta_data->uncertainty_threshold <= 0) {
                error->one(FLERR, "'uncertainty_threshold' in pair_style metatomic must be positive");
            }
            i += 1;
        } else if (strcmp(argv[i], "variant") == 0) {
            if (i == argc - 1) {
                error->one(FLERR, "expected a name after 'variant' in pair_style metatomic, got nothing");
            }
            variant = argv[i + 1];
            i += 1;
        } else if (strcmp(argv[i], "variant/energy") == 0) {
            if (i == argc - 1) {
                error->one(FLERR, "expected a name or 'off' after 'variant/energy' in pair_style metatomic, got nothing");
            }
            variant_energy = argv[i + 1];
            i += 1;
        } else if (strcmp(argv[i], "variant/energy_uncertainty") == 0) {
            if (i == argc - 1) {
                error->one(FLERR, "expected a name or 'off' after 'variant/energy_uncertainty' in pair_style metatomic, got nothing");
            }
            variant_energy_uq = argv[i + 1];
            i += 1;
        } else if (strcmp(argv[i], "variant/non_conservative_forces") == 0) {
            if (i == argc - 1) {
                error->one(FLERR, "expected a name or 'off' after 'variant/non_conservative_forces' in pair_style metatomic, got nothing");
            }
            variant_nc_forces = argv[i + 1];
            i += 1;
        } else if (strcmp(argv[i], "variant/non_conservative_stress") == 0) {
            if (i == argc - 1) {
                error->one(FLERR, "expected a name or 'off' after 'variant/non_conservative_stress' in pair_style metatomic, got nothing");
            }
            variant_nc_stress = argv[i + 1];
            i += 1;
        } else {
            error->one(FLERR, "unexpected argument to pair_style metatomic: '{}'", argv[i]);
        }
    }

    // Load the model and get its capabilities
    mta_data->load_model(this->lmp, model_path, extensions_directory);

    const auto& outputs = mta_data->capabilities->outputs();

    // Set and resolve the variants to use
    auto v_energy = variant_energy ? normalize_variant(variant_energy) : normalize_variant(variant);
    auto v_energy_uq = variant_energy_uq ? normalize_variant(variant_energy_uq) : v_energy;
    auto v_nc_forces = variant_nc_forces ? normalize_variant(variant_nc_forces) : v_energy;
    auto v_nc_stress = variant_nc_stress ? normalize_variant(variant_nc_stress) : v_energy;

    // Handle energy variant
    try {
        mta_data->energy_key = pick_output("energy", outputs, v_energy);
    } catch (std::exception& e) {
        error->one(FLERR, e.what());
    }

    // Handle energy_uncertainty variant
    if (do_uncertainty) {
        // the user did not disable energy uncertainty, let's check if the model
        // supports it
        bool has_uncertainty = false;
        for (const auto& output: outputs) {
            if (output.name().find("energy_uncertainty") == 0) {
                has_uncertainty = true;
                break;
            }
        }

        if (has_uncertainty) {
            try {
                mta_data->energy_uq_key = pick_output("energy_uncertainty", outputs, v_energy_uq);
            } catch (std::exception& e) {
                error->one(FLERR, e.what());
            }
        }
    }

    // Handle non-conservative variants
    using NCMode = PairMetatomicData::NonConservativeMode;
    const auto nc_mode = mta_data->non_conservative;

    bool user_set_forces = (variant_nc_forces != nullptr);
    bool user_set_stress = (variant_nc_stress != nullptr);

    // Warn if the user set an explicit variant for an output that the chosen
    // mode does not use.
    if (user_set_forces && nc_mode != NCMode::ON && nc_mode != NCMode::FORCES) {
        error->warning(FLERR,
            "'variant/non_conservative_forces' was set but the current 'non_conservative' mode "
            "does not use non-conservative forces; the variant will be ignored."
        );
    }
    if (user_set_stress && nc_mode != NCMode::ON && nc_mode != NCMode::STRESS) {
        error->warning(FLERR,
            "'variant/non_conservative_stress' was set but the current 'non_conservative' mode "
            "does not use non-conservative stress; the variant will be ignored."
        );
    }

    // Error if *both* nc-force and nc-stress were provided by user AND one is Null
    if (nc_mode == NCMode::ON && user_set_forces && user_set_stress) {
        bool forces_none = !normalize_variant(variant_nc_forces).has_value();
        bool stress_none = !normalize_variant(variant_nc_stress).has_value();
        if (forces_none != stress_none) {
            error->one(FLERR,
                "if both 'variant/non_conservative_stress' and "
                "'variant/non_conservative_forces' are present with 'non_conservative on', "
                "they must either both be 'off' or both not 'off'");
        }
    }

    bool do_nc_forces = (nc_mode == NCMode::ON || nc_mode == NCMode::FORCES);
    if (do_nc_forces) {
        try {
            mta_data->nc_forces_key = pick_output("non_conservative_forces", outputs, v_nc_forces);
        } catch (std::exception& e) {
            error->one(FLERR,
                "{}\nFailed to select 'non_conservative_forces' output. "
                "If the model does not support non-conservative forces, use "
                "'non_conservative stress' or 'non_conservative off'. "
                "If the model provides multiple variants, select one with "
                "'variant/non_conservative_forces <name>'.",
                e.what()
            );
        }
    }

    bool do_nc_stress = (nc_mode == NCMode::ON || nc_mode == NCMode::STRESS);
    if (do_nc_stress) {
        try {
            mta_data->nc_stress_key = pick_output("non_conservative_stress", outputs, v_nc_stress);
        } catch (std::exception& e) {
            error->one(FLERR,
                "{}\nFailed to select 'non_conservative_stress' output. "
                "If the model does not support non-conservative stress, use "
                "'non_conservative forces' or 'non_conservative off'. "
                "If the model provides multiple variants, select one with "
                "'variant/non_conservative_stress <name>'.",
                e.what()
            );
        }
    }

    // Check that the model has the required outputs
    const auto& outputs_vec = outputs;
    auto energy_output = std::find_if(outputs_vec.begin(), outputs_vec.end(),
        [&](const metatomic::Quantity& q) { return q.name() == mta_data->energy_key; });
    if (energy_output == outputs_vec.end()) {
        lmp->error->one(FLERR,
            "the model at '{}' does not have an '{}' output, "
            "we can not use it with pair_style metatomic.",
            model_path, mta_data->energy_key
        );
    }

    mta_data->is_energy_output_per_atom = energy_output->sample_kind() == metatomic::SampleKind::Atom;

    // Check the user-requested device against the model's supported devices;
    // LAMMPS itself always runs on the CPU, moving data to/from a GPU is up
    // to the model.
    mta_data->check_requested_device(lmp, requested_device, "pair_style metatomic");

    if (!allocated) {
        allocate();
    }
}

void PairMetatomic::allocate() {
    allocated = 1;

    // setflags stores whether the coeff for a given pair of atom types are known
    setflag = memory->create(
        setflag,
        atom->ntypes + 1,
        atom->ntypes + 1,
        "pair:setflag"
    );

    for (int i = 1; i <= atom->ntypes; i++) {
        for (int j = i; j <= atom->ntypes; j++) {
            setflag[i][j] = 0;
        }
    }

    // cutsq stores the squared cutoff for each pair
    cutsq = memory->create(
        cutsq,
        atom->ntypes + 1,
        atom->ntypes + 1,
        "pair:cutsq"
    );

    // lammps_types_to_species stores the mapping from lammps atom types to
    // the metatomic model species
    type_mapping = memory->create(
        type_mapping,
        atom->ntypes + 1,
        "PairMetatomic:type_mapping"
    );

    for (int i = 1; i <= atom->ntypes; i++) {
        type_mapping[i] = -1;
    }
}

double PairMetatomic::init_one(int, int) {
    return mta_data->max_cutoff;
}

// called on pair_coeff
void PairMetatomic::coeff(int argc, char ** argv) {
    if (argc < 3 || strcmp(argv[0], "*") != 0 || strcmp(argv[1], "*") != 0) {
        error->one(FLERR, "invalid pair_coeff, expected `pair_coeff * * <list of types>`");
    }

    if (atom->ntypes != argc - 2) {
        error->one(FLERR,
            "invalid pair_coeff, expected `pair_coeff * * <list of types>` with {} types",
            atom->ntypes
        );
    }

    for (int lammps_type=1; lammps_type<argc - 1; lammps_type++) {
        int type = utils::inumeric(FLERR, argv[lammps_type + 1], true, lmp);
        type_mapping[lammps_type] = type;
    }

    // mark all pairs coeffs as known
    for (int i = 1; i <= atom->ntypes; i++) {
        for (int j = 1; j <= atom->ntypes; j++) {
            setflag[i][j] = 1;
            setflag[j][i] = 1;
        }
    }
}

// called when the run starts
void PairMetatomic::init_style() {
    using NCMode = PairMetatomicData::NonConservativeMode;
    const auto nc_mode = mta_data->non_conservative;

    // Require newton pair on since we need to communicate forces accumulated on
    // ghost atoms to neighboring domains. These forces contributions come from
    // gradient of a local descriptor w.r.t. domain ghosts (periodic images
    // ghosts are handled separately).
    if (force->newton_pair != 1) {
        error->one(FLERR, "Pair style metatomic requires newton pair on");
    }

    // get the model's interaction range
    mta_data->resolve_max_cutoff(lmp);

    if (!std::isfinite(mta_data->max_cutoff)) {
        error->one(FLERR,
            "the largest cutoff of this model is infinite, "
            "we can't compute the corresponding neighbor list"
        );
    }

    // create system adaptor
    auto options = MetatomicSystemOptions{
        this->type_mapping,
        mta_data->max_cutoff,
        mta_data->check_consistency,
        nc_mode != NCMode::ON,  // autograd needed for OFF/FORCES/STRESS
    };
    this->system_adaptor = std::make_unique<MetatomicSystemAdaptor>(lmp, options);

    // We ask LAMMPS for a full neighbor lists because we need to know about
    // ALL pairs, even if options->full_list() is false. We will then filter
    // the pairs to only include each pair once where needed.
    auto request = neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_GHOST);
    this->system_adaptor->configure_neighbor_lists(request, mta_data, "pair metatomic");
}

void PairMetatomic::init_list(int id, NeighList *ptr) {
    this->mta_list = ptr;
}

void PairMetatomic::compute(int eflag, int vflag) {
    using NCMode = PairMetatomicData::NonConservativeMode;
    const auto nc_mode = mta_data->non_conservative;

    if (std::getenv("LAMMPS_METATOMIC_PROFILE") != nullptr) {
        MetatomicTimer::enable(true);
    } else {
        MetatomicTimer::enable(false);
    }

    auto _ = MetatomicTimer("PairMetatomic::compute");

    ev_init(eflag, vflag);

    // we need an energy output if the energy was explicitly requested (through
    // `eflag_either`), or when running in standard/conservative mode, because
    // we'll get the forces and stress as the gradient of the energy.
    auto need_energy_gradient = (nc_mode == NCMode::OFF
                              || nc_mode == NCMode::STRESS
                              || (nc_mode == NCMode::FORCES && vflag_global));

    auto do_nc_forces = nc_mode == NCMode::ON || nc_mode == NCMode::FORCES;
    auto do_nc_stress = nc_mode == NCMode::ON || nc_mode == NCMode::STRESS;

    // build the list of outputs to request from the model: one Quantity per
    // output, with explicit gradients on the energy when we need to derive
    // forces/virial from it.
    auto requested_outputs = std::vector<metatomic::Quantity>();
    if (eflag_either || need_energy_gradient) {
        auto energy_quantity = mta_data->energy_output;
        if (need_energy_gradient) {
            // TODO: how to add gradients to an existing Quantity? rebuild it
            // from the builder with the requested gradients
            energy_quantity = metatomic::Quantity::builder()
                .name(energy_quantity.name())
                .unit(energy_quantity.unit())
                .sample_kind(metatomic::SampleKind::System)
                .add_gradient(metatomic::Gradients::Positions)
                .add_gradient(metatomic::Gradients::Strain)
                .build();
        }
        requested_outputs.push_back(energy_quantity);
    }

    if (mta_data->uncertainty_threshold > 0) {
        requested_outputs.push_back(mta_data->uncertainty_output);
    }
    if (do_nc_forces) {
        requested_outputs.push_back(mta_data->nc_forces_output);
    }
    if (do_nc_stress) {
        requested_outputs.push_back(mta_data->nc_stress_output);
    }

    auto dtype = mta_data->model_dtype(lmp);

    // deal with the model requested inputs
    auto input_holders = mta_data->collect_requested_inputs();

    // transform from LAMMPS to metatomic System
    auto system = this->system_adaptor->system_from_lmp(
        mta_list,
        vflag_global && !do_nc_stress,
        dtype,
        input_holders
    );

    // only include atoms in the "all" group; atoms temporarily removed from
    // the all group (e.g. by fix_gcmc) are excluded.
    mta_data->set_selected_atoms(atom, 1);

    // run the model: one TensorMap per requested output, in the same order
    // as the requested_outputs vector.
    auto systems = std::vector<metatomic::System>();
    systems.push_back(std::move(system));

    auto results = std::vector<metatensor::TensorMap>();
    {
        auto _ = MetatomicTimer("running metatomic::execute_model");
        try {
            results = metatomic::execute_model(
                *mta_data->model,
                systems,
                mta_data->selected_atoms,
                requested_outputs,
                mta_data->check_consistency
            );
        } catch (const std::exception& e) {
            error->one(FLERR, "error evaluating the metatomic model: {}", e.what());
        }
    }

    if (results.empty()) {
        error->one(FLERR, "the model at '{}' returned no outputs", mta_data->model_path);
    }

    // the first requested output is always the energy (see the
    // requested_outputs construction above)
    auto& energy_map = results[0];

    {
        auto _ = MetatomicTimer("storing model output in LAMMPS data structures");

        // store the energy if requested
        if (eflag_either) {
            auto block = energy_map.block_by_id(0);
            auto values = block.values<double>();  // DLPackArray<double>

            // store the energy returned by the model
            if (eflag_atom) {
                auto samples = block.samples();
                assert(samples.count() == values.shape()[0]);

                const auto& mta_to_lmp = this->system_adaptor->mta_to_lmp;
                const auto* raw = values.data();
                for (size_t i=0; i<samples.count(); i++) {
                    auto atom_i = mta_to_lmp[raw[i]];
                    eatom[atom_i] += this->scale * raw[i];
                }
            }

            if (eflag_global) {
                auto sum = 0.0;
                const auto* raw = values.data();
                auto n = values.shape()[0];
                for (size_t i=0; i<n; i++) {
                    sum += raw[i];
                }
                eng_vdwl += this->scale * sum;
            }
        }

        // store forces: either from the NC output directly, or from the
        // gradient of the energy w.r.t. positions
        std::vector<double> forces_buffer;
        if (do_nc_forces) {
            // TODO: read from the nc_forces TensorMap
        } else if (need_energy_gradient) {
            // TODO: read gradient("positions") and copy to forces_buffer
        }
        this->store_forces(forces_buffer.data(), forces_buffer.size() / 3);

        assert(!vflag_fdotr);

        if (vflag_global) {
            // TODO: read the strain gradient / nc_stress block and fill virial
        }

        if (vflag_atom) {
            error->one(FLERR, "per atom virial is not implemented");
        }
    }
}

void PairMetatomic::store_forces(const double* forces, uintptr_t n_atoms) {
    for (uintptr_t i=0; i<n_atoms && i < static_cast<uintptr_t>(atom->nlocal); i++) {
        atom->f[i][0] += this->scale * forces[3*i + 0];
        atom->f[i][1] += this->scale * forces[3*i + 1];
        atom->f[i][2] += this->scale * forces[3*i + 2];
    }

    // ghost atom forces only exist when forces come from the energy gradient
    using NCMode = PairMetatomicData::NonConservativeMode;
    const auto nc_mode = mta_data->non_conservative;
    if (nc_mode == NCMode::OFF || nc_mode == NCMode::STRESS) {
        const auto& mta_to_lmp = this->system_adaptor->mta_to_lmp;
        for (uintptr_t i=atom->nlocal; i<n_atoms; i++) {
            auto atom_i = mta_to_lmp[i];
            atom->f[atom_i][0] += this->scale * forces[3*i + 0];
            atom->f[atom_i][1] += this->scale * forces[3*i + 1];
            atom->f[atom_i][2] += this->scale * forces[3*i + 2];
        }
    }
}
