"""Helper functions for things that are not in rdkit."""

from typing import List

from rdkit import Chem
from rdkit.Chem import Atom, Mol, RWMol


def get_atom_symbol(atom: Atom) -> str:
    """Get the chemical symbol of the given atom with the formal charge appended to it.

    Args:
        atom: An RDKit atom for which we need the symbol.

    Returns:
        The chemical symbol of the atom with the atom's formal charge appended to it as a sequence of + or - characters.

    Examples:
        >>> atom = Atom("N")
        >>> atom.SetFormalCharge(1)
        >>> get_atom_symbol(atom)
        'N+'
        >>> atom.SetFormalCharge(-2)
        >>> get_atom_symbol(atom)
        'N--'
    """
    atom_symbol = atom.GetSymbol()
    charge = atom.GetFormalCharge()
    assert isinstance(charge, int)
    charge_symbol = ""
    abs_charge = charge
    if charge < 0:
        charge_symbol = "-"
        abs_charge *= -1
    elif charge > 0:
        charge_symbol = "+"
    charge_string = charge_symbol * abs_charge
    return atom_symbol + charge_string


def get_true_symbol(symbol: str) -> str:
    """Remove trailing + or - characters from the symbol, returning just the chemical symbol.

    Examples:
        >>> get_true_symbol("N++")
        'N'
        >>> get_true_symbol("O-")
        'O'
    """
    trailing_char = symbol[-1]
    if trailing_char == "-":
        return symbol.split("-")[0]
    elif trailing_char == "+":
        return symbol.split("+")[0]
    else:
        return symbol


def get_charge_from_symbol(symbol: str) -> int:
    """Return the charge of the atom based on the symbol.

    Examples:
        >>> get_charge_from_symbol("C")
        0
        >>> get_charge_from_symbol("O-")
        -1
        >>> get_charge_from_symbol("N++")
        2
    """
    return len(symbol.split("+")) - len(symbol.split("-"))


def initialise_atom_from_symbol(symbol: str) -> Atom:
    """Initialise an RDKit atom from a symbol with a charge.

    Examples:
        >>> atom = initialise_atom_from_symbol("N+")
        >>> atom.GetSymbol()
        'N'
        >>> atom.GetFormalCharge()
        1
    """
    true_symbol = get_true_symbol(symbol)
    charge = get_charge_from_symbol(symbol)
    atom = Atom(true_symbol)
    atom.SetFormalCharge(charge)
    return atom


def compute_canonical_atom_order(mol: Mol) -> List[int]:
    """Computes the canonical ordering of atoms.

    Examples:
        >>> mol = Chem.MolFromSmiles("c1ccccc1N")
        >>> compute_canonical_atom_order(mol)
        [6, 5, 0, 1, 2, 3, 4]
        >>> canonical_smiles = Chem.MolToSmiles(mol)
        >>> canonical_smiles
        'Nc1ccccc1'
        >>> compute_canonical_atom_order(Chem.MolFromSmiles(canonical_smiles))
        [0, 1, 2, 3, 4, 5, 6]
    """
    # We need to run MolToSmiles to generate the order, but don't care about the result.
    _ = Chem.MolToSmiles(mol)

    return list(
        mol.GetPropsAsDict(includePrivate=True, includeComputed=True)["_smilesAtomOutputOrder"]
    )


def remove_atoms_outside_frag(mol: RWMol, frag_atom_ids: List[int]) -> RWMol:
    """Remove all atoms that are outside of the given list of atoms."""
    atoms_to_delete = [
        atom_idx
        # Range is reversed because atom indices in mol are recalculated every time one is removed.
        for atom_idx in reversed(range(mol.GetNumAtoms()))
        if atom_idx not in frag_atom_ids
    ]

    for atom_idx in atoms_to_delete:
        mol.RemoveAtom(atom_idx)

    return mol


if __name__ == "__main__":
    import doctest

    doctest.testmod()
