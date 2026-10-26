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
#include "metatomic_system.h"
#include "metatomic_dlpack.h"
#include "metatomic_timer.h"
#include "metatomic_types.h"
#include "metatomic_quantities.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "update.h"
#include "memory.h"

#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"

#include <map>
#include <metatensor/dlpack/dlpack.h>
#include <metatomic/metadata.hpp>
#include <metatomic/utils.hpp>
#include <string>

using namespace LAMMPS_NS;

using vector_t = std::array<double, 3>;
using matrix_t = std::array<std::array<double, 3>, 3>;

// TODO RM
constexpr DLDataType kDLFloat32{kDLFloat, 32, 1}; // float, 32 bits, non-vectorized
constexpr DLDataType kDLFloat64{kDLFloat, 64, 1}; // float, 64 bits, non-vectorized

static vector_t cross(vector_t a, vector_t b) {
    return {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    };
}

static double dot(vector_t a, vector_t b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static vector_t normalize(vector_t a) {
    double norm = std::sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
    return {a[0] / norm, a[1] / norm, a[2] / norm};
}

static double determinant(matrix_t a) {
    return a[0][0] * (a[1][1] * a[2][2] - a[2][1] * a[1][2])
         - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
         + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
}

matrix_t inverse(matrix_t a) {
    auto det = determinant(a);

    if (std::abs(det) < 1e-10) {
        throw std::runtime_error("this matrix is not invertible");
    }

    auto inverse = matrix_t();
    inverse[0][0] = (a[1][1] * a[2][2] - a[2][1] * a[1][2]) / det;
    inverse[0][1] = (a[0][2] * a[2][1] - a[0][1] * a[2][2]) / det;
    inverse[0][2] = (a[0][1] * a[1][2] - a[0][2] * a[1][1]) / det;
    inverse[1][0] = (a[1][2] * a[2][0] - a[1][0] * a[2][2]) / det;
    inverse[1][1] = (a[0][0] * a[2][2] - a[0][2] * a[2][0]) / det;
    inverse[1][2] = (a[1][0] * a[0][2] - a[0][0] * a[1][2]) / det;
    inverse[2][0] = (a[1][0] * a[2][1] - a[2][0] * a[1][1]) / det;
    inverse[2][1] = (a[2][0] * a[0][1] - a[0][0] * a[2][1]) / det;
    inverse[2][2] = (a[0][0] * a[1][1] - a[1][0] * a[0][1]) / det;
    return inverse;
}

std::array<std::array<double, 3>, 3> MetatomicSystemAdaptor::cell_inverse() {
    auto periodic = std::array<bool, 3>{
        static_cast<bool>(domain->xperiodic),
        static_cast<bool>(domain->yperiodic),
        static_cast<bool>(domain->zperiodic),
    };

    auto cell = std::array<std::array<double, 3>, 3>{{0}};
    cell[0][0] = domain->xprd;
    cell[1][0] = domain->xy;
    cell[1][1] = domain->yprd;
    cell[2][0] = domain->xz;
    cell[2][1] = domain->yz;
    cell[2][2] = domain->zprd;

    // find number of periodic directions and their indices
    int n_periodic = 0;
    int periodic_idx_1 = -1;
    int periodic_idx_2 = -1;
    for (int i = 0; i < 3; ++i) {
        if (periodic[i]) {
            n_periodic += 1;
            if (periodic_idx_1 == -1) {
                periodic_idx_1 = i;
            } else if (periodic_idx_2 == -1) {
                periodic_idx_2 = i;
            }
        }
    }

    // adjust the box matrix to have a simple orthogonal dimension along
    // non-periodic directions
    if (n_periodic == 0) {
        return {
            std::array<double, 3>{1, 0, 0},
            std::array<double, 3>{0, 1, 0},
            std::array<double, 3>{0, 0, 1},
        };
    } else if (n_periodic == 1) {
        assert(periodic_idx_1 != -1);
        // Make the two non-periodic directions orthogonal to the periodic one
        auto a = cell[periodic_idx_1];
        auto b = std::array<double, 3>{0, 1, 0};
        if (std::abs(dot(normalize(a), b)) > 0.9) {
            b = std::array<double, 3>{0, 0, 1};
        }
        auto c = normalize(cross(a, b));
        b = normalize(cross(c, a));

        // Assign back to the cell picking the "non-periodic" indices without ifs
        cell[(periodic_idx_1 + 1) % 3] = b;
        cell[(periodic_idx_1 + 2) % 3] = c;
    } else if (n_periodic == 2) {
        assert(periodic_idx_1 != -1 && periodic_idx_2 != -1);
        // Make the one non-periodic direction orthogonal to the two periodic ones
        auto a = cell[periodic_idx_1];
        auto b = cell[periodic_idx_2];
        auto c = normalize(cross(a, b));

        // Assign back to the matrix picking the "non-periodic" index without ifs
        cell[(3 - periodic_idx_1 - periodic_idx_2)] = c;
    }

    return inverse(cell);
}

MetatomicSystemAdaptor::MetatomicSystemAdaptor(LAMMPS *lmp, MetatomicSystemOptions options):
    Pointers(lmp),
    options_(std::move(options))
{
    this->strain_buffer_ = memory->create(this->strain_buffer_, 9, "metatomic:strain");
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            this->strain_buffer_[i*3+j] = (i==j) ? 1.0 : 0.0;
        }
    }
    this->strain = mta::dlpack_view(this->strain_buffer_, {3, 3});
    this->atomic_types_ = mta::dlpack_view(this->atomic_types_buffer_, {0});
}

MetatomicSystemAdaptor::~MetatomicSystemAdaptor() {
    memory->destroy(this->strain_buffer_);
    memory->destroy(this->atomic_types_buffer_);
    this->strain_buffer_ = nullptr;
    this->atomic_types_buffer_ = nullptr;
}

void MetatomicSystemAdaptor::add_nl_request(double cutoff, const metatomic::PairListOptions& request) {
    if (cutoff > options_.interaction_range) {
        error->one(FLERR,
            "Invalid metatomic model: one of the requested neighbor lists "
            "has a cutoff ({}) larger than the model interaction range ({})",
            cutoff, options_.interaction_range
        );
    } else if (cutoff < 0 || !std::isfinite(cutoff)) {
        error->one(FLERR,
            "model requested an invalid cutoff for neighbors list: {} "
            "(cutoff in model units is {})",
            cutoff, request.cutoff()
        );
    }

    nl_requests_.push_back({
        cutoff,
        request,
        /*samples = */ {},
        /*distances_f64 = */ {},
        /*distances_f32 = */ {},
    });
}

// Translate from the metatomic neighbor lists requests to LAMMPS neighbor lists requests
void MetatomicSystemAdaptor::configure_neighbor_lists(NeighRequest* request, CommonMetatomicData* mta_data, const char* requester) {
    request->set_cutoff_fixed(mta_data->max_cutoff);

    auto mincut = mta_data->max_cutoff + neighbor->skin;
    if (strcmp(requester, "pair metatomic") != 0 && comm->get_comm_cutoff() < mincut) {
        if (comm->me == 0) {
            error->warning(FLERR,
                "Increasing communication cutoff to {:.8} for {}",
                mincut, requester
            );
        }
        comm->cutghostuser = mincut;
    }

    // Translate from the metatomic neighbor lists requests to LAMMPS neighbor
    // lists requests.
    auto requested_nl = mta_data->model->requested_pair_lists();
    for (const auto& pair_list_options: requested_nl) {
        // TODO RM
        //auto cutoff = options->engine_cutoff(mta_data->evaluation_options->length_unit());
        auto cutoff = pair_list_options.cutoff();
        assert(cutoff <= mta_data->max_cutoff);

        this->add_nl_request(cutoff, pair_list_options);
    }

}

static std::array<int32_t, 3> cell_shifts(
    const std::array<std::array<double, 3>, 3>& cell_inv,
    const std::array<double, 3>& pair_shift
) {
    auto shift_a = static_cast<int32_t>(std::round(
        cell_inv[0][0] * pair_shift[0] +
        cell_inv[1][0] * pair_shift[1] +
        cell_inv[2][0] * pair_shift[2]
    ));
    auto shift_b = static_cast<int32_t>(std::round(
        cell_inv[0][1] * pair_shift[0] +
        cell_inv[1][1] * pair_shift[1] +
        cell_inv[2][1] * pair_shift[2]
    ));
    auto shift_c = static_cast<int32_t>(std::round(
        cell_inv[0][2] * pair_shift[0] +
        cell_inv[1][2] * pair_shift[1] +
        cell_inv[2][2] * pair_shift[2]
    ));

    return {shift_a, shift_b, shift_c};
}

template <typename T>
static metatensor::TensorMap make_per_atom_tensormap(
    T *data,
    std::vector<uintptr_t> shape,
    const std::string &property_name,
    const std::vector<std::string> &component_names
) {
    assert (shape.size() == component_names.size() + 1);

    auto n_atoms = shape[0];

    auto samples_values = std::vector<int32_t>(2 * n_atoms);
    for (uintptr_t i = 0; i < n_atoms; i++) {
        samples_values[2 * i] = 0;
        samples_values[2 * i + 1] = static_cast<int32_t>(i);
    }
    auto samples = metatensor::Labels(
        std::vector<std::string>{"system", "atom"},
        samples_values.data(), static_cast<size_t>(n_atoms),
        metatensor::assume_unique{}
    );

    auto components = std::vector<metatensor::Labels>();
    for (size_t axis = 0; axis < component_names.size(); axis++) {
        auto m = shape[axis + 1];
        auto entries = std::vector<int32_t>(m);
        for (uintptr_t i = 0; i < m; i++) {
            entries[i] = static_cast<int32_t>(i);
        }
        components.push_back(metatensor::Labels(
            std::vector<std::string>{component_names[axis]},
            entries.data(), static_cast<size_t>(m)
        ));
    }

    auto properties = metatensor::Labels(
        std::vector<std::string>{std::move(property_name)}, {{0}}
    );

    shape.push_back(1);
    auto values = std::unique_ptr<metatensor::DataArrayBase>(
        std::make_unique<mta::LmpDataArray<T>>(data, std::move(shape))
    );

    auto block = metatensor::TensorBlock(
        std::move(values), samples, components, properties
    );

    auto blocks = std::vector<metatensor::TensorBlock>();
    blocks.push_back(std::move(block));

    return metatensor::TensorMap(
        // LabelsHolder::single()
        metatensor::Labels(std::vector<std::string>{"_"}, {{0}}),
        std::move(blocks)
    );
}

void MetatomicSystemAdaptor::guess_periodic_ghosts() {
    auto _ = MetatomicTimer("identifying periodic ghosts");
    auto total_n_atoms = atom->nlocal + atom->nghost;
    double** x = atom->x;

    original_atom_id_.clear();
    original_atom_id_.reserve(total_n_atoms);
    lmp_to_mta_.clear();
    lmp_to_mta_.reserve(total_n_atoms);
    mta_to_lmp.clear();
    mta_to_lmp.reserve(total_n_atoms);

    local_atoms_tags_.clear();
    for (int i = 0; i < atom->nlocal; i++) {
        local_atoms_tags_.emplace(atom->tag[i], i);
        original_atom_id_.emplace_back(i);
        lmp_to_mta_.emplace_back(mta_to_lmp.size());
        mta_to_lmp.emplace_back(i);
    }

    // Subdomain center: used as the reference point for deterministic
    // representative selection. Among all ghosts sharing a tag, the one
    // closest to the subdomain center is chosen. This is deterministic
    // regardless of ghost array ordering (which is non-deterministic on
    // GPU) because ghost positions depend only on the original atom
    // position and exact cell-vector shifts — both of which are
    // order-independent. Picking the closest ghost also gives the most
    // natural representative for cell-shift calculations.
    double center[3];
    if (domain->triclinic == 0) {
        center[0] = 0.5 * (domain->sublo[0] + domain->subhi[0]);
        center[1] = 0.5 * (domain->sublo[1] + domain->subhi[1]);
        center[2] = 0.5 * (domain->sublo[2] + domain->subhi[2]);
    } else {
        // For triclinic boxes, domain->sublo/subhi are not set (see
        // Domain::set_local_box, which returns early). The subdomain
        // bounds are stored in reduced coordinates instead; since
        // lamda2x is affine, the midpoint in lamda maps onto the
        // center of the Cartesian parallelepiped.
        double center_lamda[3] = {
            0.5 * (domain->sublo_lamda[0] + domain->subhi_lamda[0]),
            0.5 * (domain->sublo_lamda[1] + domain->subhi_lamda[1]),
            0.5 * (domain->sublo_lamda[2] + domain->subhi_lamda[2])
        };
        domain->lamda2x(center_lamda, center);
    }

    // We do the first pass over ghost atoms to find the representative for
    // for each tag, then a second pass to build the mapping arrays. This ensures
    // that the representative selection is not affected by the order of ghost
    // atoms in the arrays, which can be non-deterministic on GPU.

    ghost_atoms_tags_.clear();
    for (int i = atom->nlocal; i < total_n_atoms; i++) {
        auto tag = atom->tag[i];
        if (local_atoms_tags_.count(tag)) {
            continue;  // owned locally, local atom is the representative
        }

        auto it = ghost_atoms_tags_.find(tag);
        if (it == ghost_atoms_tags_.end()) {
            ghost_atoms_tags_.emplace(tag, i);
        } else {
            // Replace if the new ghost is closer to the subdomain center.
            double dist_new = 0, dist_old = 0;
            for (int d = 0; d < 3; d++) {
                double dn = x[i][d] - center[d];
                double de = x[it->second][d] - center[d];
                dist_new += dn * dn;
                dist_old += de * de;
            }
            if (dist_new < dist_old) {
                it->second = i;
            } else if (dist_new == dist_old) {
                // Lexicographic tiebreaker for the rare equal-distance case
                for (int d = 0; d < 3; d++) {
                    if (x[i][d] < x[it->second][d]) {
                        it->second = i;
                        break;
                    } else if (x[i][d] > x[it->second][d]) {
                        break;
                    }
                }
            }
        }
    }

    // Second pass over the ghost atoms to build the mapping arrays,
    // since now we know which one is the representative for each tag.
    for (int i = atom->nlocal; i < total_n_atoms; i++) {
        auto tag = atom->tag[i];
        auto local_it = local_atoms_tags_.find(tag);
        if (local_it != local_atoms_tags_.end()) {
            // periodic image of a local atom
            original_atom_id_.emplace_back(local_it->second);
            lmp_to_mta_.emplace_back(-1);
        } else {
            auto ghost_it = ghost_atoms_tags_.find(tag);
            auto representative = ghost_it->second;
            original_atom_id_.emplace_back(representative);
            if (representative == i) {
                // this ghost IS the representative
                lmp_to_mta_.emplace_back(mta_to_lmp.size());
                mta_to_lmp.emplace_back(i);
            } else {
                // periodic image of an inter-domain ghost
                lmp_to_mta_.emplace_back(-1);
            }
        }
    }
}


void MetatomicSystemAdaptor::setup_neighbors(metatomic::System& system, NeighList *list) {
    auto _ = MetatomicTimer("converting neighbors list");
    auto dtype = system.positions()->dl_tensor.dtype;
    auto device = system.types()->dl_tensor.device;

    double** x = atom->x;
    auto cell_inv = this->cell_inverse();

    for (auto& nl: nl_requests_) {
        {
            auto _ = MetatomicTimer("filtering LAMMPS neighbor list");

            auto cutoff2 = nl.cutoff * nl.cutoff;
            auto full_list = nl.options.full_list();

            // convert from LAMMPS neighbors list to metatomic format
            nl.samples.clear();
            nl.distances_f32.clear();
            nl.distances_f64.clear();
            for (int ii=0; ii<(list->inum + list->gnum); ii++) {
                auto atom_i = list->ilist[ii];
                auto original_atom_i = original_atom_id_[atom_i];
                auto i_is_original = (atom_i == original_atom_i);

                auto neighbors = list->firstneigh[atom_i];
                for (int jj=0; jj<list->numneigh[atom_i]; jj++) {
                    auto atom_j = neighbors[jj] & NEIGHMASK;
                    auto original_atom_j = original_atom_id_[atom_j];
                    auto j_is_original = (atom_j == original_atom_j);

                    if (!full_list && original_atom_i > original_atom_j) {
                        // Remove extra pairs if the model requested half-lists
                        continue;
                    }

                    if (!i_is_original && !j_is_original) {
                        // both atoms are periodic ghosts, skip the pair
                        continue;
                    }

                    if (!i_is_original && j_is_original) {
                        // this pair will be accounted for when we will process
                        // atom_j as the central atom
                        continue;
                    }

                    auto distance = std::array<double, 3>{
                        x[atom_j][0] - x[atom_i][0],
                        x[atom_j][1] - x[atom_i][1],
                        x[atom_j][2] - x[atom_i][2],
                    };

                    auto distance2 = (
                        distance[0] * distance[0] +
                        distance[1] * distance[1] +
                        distance[2] * distance[2]
                    );
                    if (distance2 > cutoff2) {
                        // LAMMPS neighbors list contains some pairs after the
                        // cutoff, we filter them here
                        continue;
                    }

                    // Compute the cell shift for the pair.
                    auto shift_i = std::array<double, 3>{
                        x[atom_i][0] - x[original_atom_i][0],
                        x[atom_i][1] - x[original_atom_i][1],
                        x[atom_i][2] - x[original_atom_i][2],
                    };
                    auto shift_j = std::array<double, 3>{
                        x[atom_j][0] - x[original_atom_j][0],
                        x[atom_j][1] - x[original_atom_j][1],
                        x[atom_j][2] - x[original_atom_j][2],
                    };
                    auto pair_shift = std::array<double, 3>{
                        shift_j[0] - shift_i[0],
                        shift_j[1] - shift_i[1],
                        shift_j[2] - shift_i[2],
                    };

                    auto shift = std::array<int32_t, 3>{0, 0, 0};
                    if (pair_shift[0] != 0 || pair_shift[1] != 0 || pair_shift[2] != 0) {
                        shift = cell_shifts(cell_inv, pair_shift);

                        if (!full_list && original_atom_i == original_atom_j) {
                            // If a half neighbors list has been requested, do
                            // not include the same pair between an atom and
                            // it's periodic image twice with opposite cell
                            // shifts (e.g. [1, -1, 1] and [-1, 1, -1]).
                            //
                            // Instead we pick pairs in the positive plan of
                            // shifts.
                            if (shift[0] + shift[1] + shift[2] < 0) {
                                // drop shifts on the negative half-space
                                continue;
                            }

                            if ((shift[0] + shift[1] + shift[2] == 0)
                                && (shift[2] < 0 || (shift[2] == 0 && shift[1] < 0))) {
                                // drop shifts in the negative half plane or the
                                // negative shift[1] axis.
                                //
                                // See below for a graphical representation: we are
                                // keeping the shifts indicated with `O` and
                                // dropping the ones indicated with `X`
                                //
                                //  O O O │ O O O
                                //  O O O │ O O O
                                //  O O O │ O O O
                                // ─X─X─X─┼─O─O─O─
                                //  X X X │ X X X
                                //  X X X │ X X X
                                //  X X X │ X X X
                                continue;
                            }
                        }
                    }

                    auto sample = std::array<int32_t, 5>{
                        lmp_to_mta_[original_atom_i],
                        lmp_to_mta_[original_atom_j],
                        shift[0],
                        shift[1],
                        shift[2],
                    };
                    nl.samples.push_back(sample);
                    if (dtype.code == kDLFloat && dtype.bits == 64 && dtype.lanes == 1) { // TODO RM
                        nl.distances_f64.push_back(distance);
                    } else if (dtype.code == kDLFloat && dtype.bits == 32 && dtype.lanes == 1) { // TODO RM
                        nl.distances_f32.push_back({
                            static_cast<float>(distance[0]),
                            static_cast<float>(distance[1]),
                            static_cast<float>(distance[2])
                        });
                    } else {
                        // should be unreachable
                        error->one(FLERR, "invalid dtype, this is a bug");
                    }
                }
            }
        }

        int64_t n_pairs = nl.samples.size();
        auto dimensions = std::vector<std::string>{
            "first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"
        };
        auto values = reinterpret_cast<const int32_t*>(nl.samples.data());
        auto samples = options_.check_consistency
            ? metatensor::Labels(dimensions, values, n_pairs)
            : metatensor::Labels(dimensions, values, n_pairs, metatensor::assume_unique{});

        std::unique_ptr<metatensor::DataArrayBase> distances_vectors;
        if (dtype.code == kDLFloat && dtype.bits == 64 && dtype.lanes == 1) { // TODO RM
            distances_vectors = std::unique_ptr<metatensor::DataArrayBase>(
                new mta::LmpDataArray<double>(
                    reinterpret_cast<double*>(nl.distances_f64.data()),
                    std::vector<uintptr_t>{n_pairs, 3, 1}
                )
            );
        } else if (dtype.code == kDLFloat && dtype.bits == 32 && dtype.lanes == 1) { // TODO RM
            distances_vectors = std::unique_ptr<metatensor::DataArrayBase>(
                new mta::LmpDataArray<float>(
                    reinterpret_cast<float*>(nl.distances_f32.data()),
                    std::vector<uintptr_t>{n_pairs, 3, 1}
                )
            );
        } else {
            // should be unreachable
            error->one(FLERR, "invalid dtype, this is a bug");
        }

        auto neighbors = metatensor::TensorBlock(
            std::move(distances_vectors), samples,
            std::vector<metatensor::Labels>{metatensor::Labels({"xyz"}, {{0}, {1}, {2}})},
            metatensor::Labels({"distance"}, {{0}})
        );

        // TODO RM
        //metatomic_torch::register_autograd_neighbors(system, neighbors, options_.check_consistency);
        system.add_pairs(nl.options, std::move(neighbors));
    }
}

metatomic::System MetatomicSystemAdaptor::system_from_lmp(
    NeighList* list,
    bool do_virial,
    DLDataType dtype,
    DLDeviceType device,
    const std::map<std::string, metatomic::Quantity>& requested_inputs
) {
    auto _ = MetatomicTimer("creating System from LAMMPS data");

    double** x = atom->x;
    auto total_n_atoms = atom->nlocal + atom->nghost;

    // atomic types for all atoms (local + ghost), indexed by LAMMPS atom id
    memory->grow(atomic_types_buffer_, total_n_atoms, "metatomic:atomic_types");
    for (int i=0; i<total_n_atoms; i++) {
        atomic_types_buffer_[i] = options_.types_mapping[atom->type[i]];
    }

    // build the mta_to_lmp mapping (which atoms are included in the system)
    this->guess_periodic_ghosts();

    // gather the positions and atomic types of the atoms included in the
    // metatomic system (i.e. dropping periodic-image duplicates)
    auto n_system_atoms = static_cast<int>(mta_to_lmp.size());
    memory->grow(positions_buffer_, 3 * n_system_atoms, "metatomic:positions");
    for (int i=0; i<n_system_atoms; i++) {
        auto j = mta_to_lmp[i];
        positions_buffer_[3*i]   = x[j][0];
        positions_buffer_[3*i+1] = x[j][1];
        positions_buffer_[3*i+2] = x[j][2];
    }
    // mta_to_lmp is strictly increasing, so this in-place subset gather only
    // reads entries that have not been overwritten yet
    for (int i=0; i<n_system_atoms; i++) {
        atomic_types_buffer_[i] = atomic_types_buffer_[mta_to_lmp[i]];
    }

    // views over the LAMMPS-owned double buffers; these keep the LAMMPS data
    // (before any dtype conversion) for gradient computations
    this->positions = mta::dlpack_view(positions_buffer_, {n_system_atoms, 3});
    this->atomic_types_ = mta::dlpack_view(atomic_types_buffer_, {n_system_atoms});

    // cell matrix of the simulation box, row-major 3x3 with the cell vectors
    // as rows: a = (xprd, 0, 0), b = (xy, yprd, 0), c = (xz, yz, zprd)
    cell_buffer_[0] = domain->xprd;
    cell_buffer_[3] = domain->xy;
    cell_buffer_[4] = domain->yprd;
    cell_buffer_[6] = domain->xz;
    cell_buffer_[7] = domain->yz;
    cell_buffer_[8] = domain->zprd;

    pbc_buffer_[0] = domain->xperiodic;
    pbc_buffer_[1] = domain->yperiodic;
    pbc_buffer_[2] = domain->zperiodic;

    // zero the cell rows of non-periodic directions
    if (!pbc_buffer_[0]) {
        cell_buffer_[0] = 0.0;
        cell_buffer_[1] = 0.0;
        cell_buffer_[2] = 0.0;
    }
    if (!pbc_buffer_[1]) {
        cell_buffer_[3] = 0.0;
        cell_buffer_[4] = 0.0;
        cell_buffer_[5] = 0.0;
    }
    if (!pbc_buffer_[2]) {
        cell_buffer_[6] = 0.0;
        cell_buffer_[7] = 0.0;
        cell_buffer_[8] = 0.0;
    }

    // build the views passed to the System, converting to the dtype requested
    // by the model if needed. TODO RM: expand to GPU devices
    metatomic::DLPackTensor types_tensor;
    metatomic::DLPackTensor positions_tensor;
    metatomic::DLPackTensor cell_tensor = mta::dlpack_view(cell_buffer_, {3, 3});
    metatomic::DLPackTensor pbc_tensor = mta::dlpack_view(pbc_buffer_, {3});
    if (dtype.code == kDLFloat && dtype.bits == 64 && dtype.lanes == 1) {
        types_tensor = mta::dlpack_view(atomic_types_buffer_, {n_system_atoms});
        positions_tensor = mta::dlpack_view(positions_buffer_, {n_system_atoms, 3});
    } else if (dtype.code == kDLFloat && dtype.bits == 32 && dtype.lanes == 1) {
        memory->grow(positions_f32_buffer_, 3 * n_system_atoms, "metatomic:positions_f32");
        for (int i=0; i<3*n_system_atoms; i++) {
            positions_f32_buffer_[i] = static_cast<float>(positions_buffer_[i]);
        }
        for (int i=0; i<9; i++) {
            cell_f32_buffer_[i] = static_cast<float>(cell_buffer_[i]);
        }
        types_tensor = mta::dlpack_view(atomic_types_buffer_, {n_system_atoms});
        positions_tensor = mta::dlpack_view(positions_f32_buffer_, {n_system_atoms, 3});
        cell_tensor = mta::dlpack_view(cell_f32_buffer_, {3, 3});
    } else {
        error->one(FLERR, "invalid dtype, this is a bug");
    }

    // TODO RM
    (void) do_virial;

    auto system = metatomic::System(
        metatomic_unit_map.at("position").at(update->unit_style),
        std::move(types_tensor),
        std::move(positions_tensor),
        std::move(cell_tensor),
        std::move(pbc_tensor)
    );

    this->setup_neighbors(system, list);

    for (const auto& [name, input]: requested_inputs) {
        if (name == "mass" || name == "masses") {
            add_masses(system, name);
        } else if (name == "momentum" || name == "momenta") {
            add_momenta(system, name);
        } else if (name == "velocity" || name == "velocities") {
            add_velocities(system, name);
        } else {
            error->all(FLERR, "compute metatomic: the model requested an unsupported additional input '{}'", name);
        }
    }

    return system;
}

void MetatomicSystemAdaptor::add_masses(metatomic::System& system, std::string name) {
    double* rmass = atom->rmass;
    double* mass = atom->mass;
    int* type = atom->type;

    auto n_system_atoms = static_cast<int>(mta_to_lmp.size());
    auto dtype = system.positions()->dl_tensor.dtype;

    // gather the masses of the atoms included in the metatomic system,
    // converting to the system dtype on the fly. TODO RM: expand to GPU
    if (dtype.code == kDLFloat && dtype.bits == 64 && dtype.lanes == 1) {
        memory->grow(masses_f64_buffer_, n_system_atoms, "metatomic:masses_f64");
        for (int i=0; i<n_system_atoms; i++) {
            auto j = mta_to_lmp[i];
            masses_f64_buffer_[i] = rmass ? rmass[j] : mass[type[j]];
        }
        auto tensor = make_per_atom_tensormap(
            masses_f64_buffer_, {static_cast<uintptr_t>(n_system_atoms)}, "mass", {}
        );
        auto unit = metatomic_unit_map.at("mass").at(update->unit_style);
        tensor.set_info("unit", unit);
        assert(name == "mass" || name == "masses");
        system.add_custom_data(name, std::move(tensor));
    } else if (dtype.code == kDLFloat && dtype.bits == 32 && dtype.lanes == 1) {
        memory->grow(masses_f32_buffer_, n_system_atoms, "metatomic:masses_f32");
        for (int i=0; i<n_system_atoms; i++) {
            auto j = mta_to_lmp[i];
            masses_f32_buffer_[i] = static_cast<float>(rmass ? rmass[j] : mass[type[j]]);
        }
        auto tensor = make_per_atom_tensormap(
            masses_f32_buffer_, {static_cast<uintptr_t>(n_system_atoms)}, "mass", {}
        );
        auto unit = metatomic_unit_map.at("mass").at(update->unit_style);
        tensor.set_info("unit", unit);
        assert(name == "mass" || name == "masses");
        system.add_custom_data(name, std::move(tensor));
    } else {
        error->one(FLERR, "invalid dtype, this is a bug");
    }
}


void MetatomicSystemAdaptor::add_momenta(metatomic::System& system, std::string name) {
    double* rmass = atom->rmass;
    double* mass = atom->mass;
    double** v = atom->v;
    int* type = atom->type;

    auto n_system_atoms = static_cast<int>(mta_to_lmp.size());
    auto dtype = system.types()->dl_tensor.dtype;

    // gather the momenta of the atoms included in the metatomic system,
    // converting to the system dtype on the fly. TODO RM: expand to GPU
    if (dtype.code == kDLFloat && dtype.bits == 64 && dtype.lanes == 1) {
        memory->grow(momenta_f64_buffer_, 3 * n_system_atoms, "metatomic:momenta_f64");
        for (int i=0; i<n_system_atoms; i++) {
            auto j = mta_to_lmp[i];
            double m = rmass ? rmass[j] : mass[type[j]];
            momenta_f64_buffer_[3*i]   = m * v[j][0];
            momenta_f64_buffer_[3*i+1] = m * v[j][1];
            momenta_f64_buffer_[3*i+2] = m * v[j][2];
        }
        auto tensor = make_per_atom_tensormap(
            momenta_f64_buffer_,
            {static_cast<uintptr_t>(n_system_atoms), 3},
            "momentum", {"xyz"}
        );
        auto mass_unit = metatomic_unit_map.at("mass").at(update->unit_style);
        auto velocity_unit = metatomic_unit_map.at("velocity").at(update->unit_style);
        tensor.set_info("unit", mass_unit + "*" + velocity_unit);
        assert(name == "momentum" || name == "momenta");
        system.add_custom_data(name, std::move(tensor));
    } else if (dtype.code == kDLFloat && dtype.bits == 32 && dtype.lanes == 1) {
        memory->grow(momenta_f32_buffer_, 3 * n_system_atoms, "metatomic:momenta_f32");
        for (int i=0; i<n_system_atoms; i++) {
            auto j = mta_to_lmp[i];
            double m = rmass ? rmass[j] : mass[type[j]];
            momenta_f32_buffer_[3*i]   = static_cast<float>(m * v[j][0]);
            momenta_f32_buffer_[3*i+1] = static_cast<float>(m * v[j][1]);
            momenta_f32_buffer_[3*i+2] = static_cast<float>(m * v[j][2]);
        }
        auto tensor = make_per_atom_tensormap(
            momenta_f32_buffer_,
            {static_cast<uintptr_t>(n_system_atoms), 3},
            "momentum", {"xyz"}
        );
        auto mass_unit = metatomic_unit_map.at("mass").at(update->unit_style);
        auto velocity_unit = metatomic_unit_map.at("velocity").at(update->unit_style);
        tensor.set_info("unit", mass_unit + "*" + velocity_unit);
        assert(name == "momentum" || name == "momenta");
        system.add_custom_data(name, std::move(tensor));
    } else {
        error->one(FLERR, "invalid dtype, this is a bug");
    }
}

void MetatomicSystemAdaptor::add_velocities(metatomic::System& system, std::string name) {
    double** v = atom->v;

    auto n_system_atoms = static_cast<int>(mta_to_lmp.size());
    auto dtype = system.types()->dl_tensor.dtype;

    // gather the velocities of the atoms included in the metatomic system,
    // converting to the system dtype on the fly. TODO RM: expand to GPU
    if (dtype.code == kDLFloat && dtype.bits == 64 && dtype.lanes == 1) {
        memory->grow(velocities_f64_buffer_, 3 * n_system_atoms, "metatomic:velocities_f64");
        for (int i=0; i<n_system_atoms; i++) {
            auto j = mta_to_lmp[i];
            velocities_f64_buffer_[3*i]   = v[j][0];
            velocities_f64_buffer_[3*i+1] = v[j][1];
            velocities_f64_buffer_[3*i+2] = v[j][2];
        }
        auto tensor = make_per_atom_tensormap(
            velocities_f64_buffer_,
            {static_cast<uintptr_t>(n_system_atoms), 3},
            "velocity", {"xyz"}
        );
        auto unit = metatomic_unit_map.at("velocity").at(update->unit_style);
        tensor.set_info("unit", unit);
        assert(name == "velocity" || name == "velocities");
        system.add_custom_data(name, std::move(tensor));
    } else if (dtype.code == kDLFloat && dtype.bits == 32 && dtype.lanes == 1) {
        memory->grow(velocities_f32_buffer_, 3 * n_system_atoms, "metatomic:velocities_f32");
        for (int i=0; i<n_system_atoms; i++) {
            auto j = mta_to_lmp[i];
            velocities_f32_buffer_[3*i]   = static_cast<float>(v[j][0]);
            velocities_f32_buffer_[3*i+1] = static_cast<float>(v[j][1]);
            velocities_f32_buffer_[3*i+2] = static_cast<float>(v[j][2]);
        }
        auto tensor = make_per_atom_tensormap(
            velocities_f32_buffer_,
            {static_cast<uintptr_t>(n_system_atoms), 3},
            "velocity", {"xyz"}
        );
        auto unit = metatomic_unit_map.at("velocity").at(update->unit_style);
        tensor.set_info("unit", unit);
        assert(name == "velocity" || name == "velocities");
        system.add_custom_data(name, std::move(tensor));
    } else {
        error->one(FLERR, "invalid dtype, this is a bug");
    }
}
