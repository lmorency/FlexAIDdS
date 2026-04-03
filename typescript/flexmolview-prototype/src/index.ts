export type AtomRecord = {
  serial: number;
  name: string;
  resname: string;
  chainId: string;
  resseq: number;
  x: number;
  y: number;
  z: number;
  record: "ATOM" | "HETATM";
  element: string;
};

export type Molecule = {
  atoms: AtomRecord[];
};

export function parsePDB(text: string): Molecule {
  const atoms: AtomRecord[] = [];
  for (const line of text.split(/\r?\n/)) {
    if (!(line.startsWith("ATOM") || line.startsWith("HETATM"))) continue;
    const slice = (start: number, end: number) => line.slice(start, end).trim();
    atoms.push({
      serial: Number(slice(6, 11)) || 0,
      name: slice(12, 16),
      resname: slice(17, 20),
      chainId: slice(21, 22),
      resseq: Number(slice(22, 26)) || 0,
      x: Number(slice(30, 38)) || 0,
      y: Number(slice(38, 46)) || 0,
      z: Number(slice(46, 54)) || 0,
      record: line.startsWith("HETATM") ? "HETATM" : "ATOM",
      element: slice(76, 78),
    });
  }
  return { atoms };
}

export function select(molecule: Molecule, expression: string): number[] {
  const trimmed = expression.trim();
  if (!trimmed) return [];
  const tokens = trimmed.split(/\s+/);
  if (tokens.length === 1) return selectSimple(molecule, tokens[0]);
  if (tokens.length === 2) return selectKeyed(molecule, tokens[0], tokens[1]);
  throw new Error(`Unsupported selection expression: ${expression}`);
}

function selectSimple(molecule: Molecule, token: string): number[] {
  switch (token.toLowerCase()) {
    case "all":
      return molecule.atoms.map((_, i) => i);
    case "polymer":
      return molecule.atoms.flatMap((atom, i) => (atom.record === "ATOM" ? [i] : []));
    case "ligand":
      return molecule.atoms.flatMap((atom, i) => {
        const resn = atom.resname.toUpperCase();
        return atom.record === "HETATM" && resn !== "HOH" && resn !== "ZN" ? [i] : [];
      });
    default:
      return [];
  }
}

function selectKeyed(molecule: Molecule, key: string, value: string): number[] {
  switch (key.toLowerCase()) {
    case "chain":
      return molecule.atoms.flatMap((atom, i) => (atom.chainId.toUpperCase() === value.toUpperCase() ? [i] : []));
    case "resn":
      return molecule.atoms.flatMap((atom, i) => (atom.resname.toUpperCase() === value.toUpperCase() ? [i] : []));
    case "resi": {
      const target = Number(value);
      return molecule.atoms.flatMap((atom, i) => (atom.resseq === target ? [i] : []));
    }
    case "name":
      return molecule.atoms.flatMap((atom, i) => (atom.name.toUpperCase() === value.toUpperCase() ? [i] : []));
    default:
      throw new Error(`Unsupported selection token: ${key}`);
  }
}

export class FlexMolViewPrototype {
  private molecule: Molecule = { atoms: [] };

  loadPDB(text: string): void {
    this.molecule = parsePDB(text);
  }

  atomCount(): number {
    return this.molecule.atoms.length;
  }

  select(expression: string): number[] {
    return select(this.molecule, expression);
  }
}
