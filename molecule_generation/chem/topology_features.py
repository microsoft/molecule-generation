"""Functions to constrain edge choices based on molecule topology."""
import logging
from typing import Collection

import numpy as np
from rdkit.Chem import BondType, GetSSSR, RWMol

logger = logging.getLogger(__name__)


def calculate_topology_features(edges: Collection, mol: RWMol) -> np.ndarray:
    """Constrain edges based on how many loops would be in a resulting molecule.

    Args:
        edges: a collection of shape (E, 2) representing edges that we might want to add to the molecule supplied. Each
            edge must have element 0 containing the source atom index of the edge, and element 1 containing the target
            atom index.
        mol: an RDKit read-write molecule to which we want to add at least one of the edges in the given list of edges.

    Returns:
        An int32 numpy array topology_features of shape [E, 2]. Here,
        topology_features[i, 0] is the number of rings created by adding the corresponding edge
            in the edges list, calculated as the difference in size of the smallest set of smallest
            rings (SSSR) in the molecule before and after the edge is added; and
        topology_features[i, 1] is the number of "tri-ring" edges added thanks to the edge in the
            given list, where a "tri-ring" edge is a edge that is involved in (at least) three rings
            in the SSSR.
    """
    # Make sure sensible default values are here in case RDKit throws an error.
    num_rings_created_by_edge = np.zeros(shape=(len(edges),), dtype=np.float32)
    num_tri_ring_edges_created_by_edge = np.zeros_like(num_rings_created_by_edge)
    mol_copy = RWMol(mol)
    try:
        # Must be calculated before GetRingInfo is called, to ensure ring info is initialised.
        num_rings_in_base_mol = GetSSSR(mol_copy)
        num_base_tri_ring_edges = _calculate_num_tri_rings(mol_copy)

        for edge_idx, edge in enumerate(edges):
            test_mol = RWMol(mol)
            test_mol.AddBond(int(edge[0]), int(edge[1]), BondType.SINGLE)
            num_rings_with_new_edge = GetSSSR(test_mol)
            num_tri_ring_edges = _calculate_num_tri_rings(test_mol)

            num_tri_ring_edges_created_by_edge[edge_idx] = (
                num_tri_ring_edges - num_base_tri_ring_edges
            )
            num_rings_created_by_edge[edge_idx] = num_rings_with_new_edge - num_rings_in_base_mol

        return np.stack([num_rings_created_by_edge, num_tri_ring_edges_created_by_edge], axis=-1)

    except Exception as e:
        logger.warning("RDKit runtime error on base molecule, with message:\n" + str(e))
        return np.stack([num_rings_created_by_edge, num_tri_ring_edges_created_by_edge], axis=-1)


def _calculate_num_tri_rings(mol: RWMol) -> int:
    ring_info = mol.GetRingInfo()
    return sum(ring_info.NumBondRings(bond_idx) >= 3 for bond_idx in range(mol.GetNumBonds()))
