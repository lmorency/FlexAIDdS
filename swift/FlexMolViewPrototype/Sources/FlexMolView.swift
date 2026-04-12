import Foundation

public struct FMVAtom: Sendable, Equatable {
    public let serial: Int
    public let name: String
    public let resname: String
    public let chainID: String
    public let resseq: Int
    public let x: Double
    public let y: Double
    public let z: Double
    public let record: String
    public let element: String

    public init(serial: Int, name: String, resname: String, chainID: String, resseq: Int, x: Double, y: Double, z: Double, record: String, element: String) {
        self.serial = serial
        self.name = name
        self.resname = resname
        self.chainID = chainID
        self.resseq = resseq
        self.x = x
        self.y = y
        self.z = z
        self.record = record
        self.element = element
    }
}

public struct FMVMolecule: Sendable, Equatable {
    public let atoms: [FMVAtom]

    public init(atoms: [FMVAtom]) {
        self.atoms = atoms
    }

    public var atomCount: Int { atoms.count }
}

public enum FMVSelectionError: Error, Equatable {
    case invalidExpression(String)
}

public enum FMVParser {
    public static func parsePDB(_ text: String) -> FMVMolecule {
        let atoms = text.split(whereSeparator: \ .isNewline).compactMap { raw -> FMVAtom? in
            let line = String(raw)
            guard line.hasPrefix("ATOM") || line.hasPrefix("HETATM") else { return nil }
            func slice(_ start: Int, _ end: Int) -> String {
                guard start < line.count else { return "" }
                let s = line.index(line.startIndex, offsetBy: max(0, start))
                let e = line.index(line.startIndex, offsetBy: min(line.count, end))
                return String(line[s..<e]).trimmingCharacters(in: .whitespaces)
            }
            return FMVAtom(
                serial: Int(slice(6, 11)) ?? 0,
                name: slice(12, 16),
                resname: slice(17, 20),
                chainID: slice(21, 22),
                resseq: Int(slice(22, 26)) ?? 0,
                x: Double(slice(30, 38)) ?? 0,
                y: Double(slice(38, 46)) ?? 0,
                z: Double(slice(46, 54)) ?? 0,
                record: slice(0, 6),
                element: slice(76, 78)
            )
        }
        return FMVMolecule(atoms: atoms)
    }
}

public enum FMVSelection {
    public static func select(_ expression: String, in molecule: FMVMolecule) throws -> [Int] {
        let trimmed = expression.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { throw FMVSelectionError.invalidExpression("empty selection") }
        let tokens = trimmed.split(separator: " ").map(String.init)
        if tokens.count == 1 {
            return simple(tokens[0], in: molecule)
        }
        if tokens.count == 2 {
            return try keyed(tokens[0], value: tokens[1], in: molecule)
        }
        throw FMVSelectionError.invalidExpression(expression)
    }

    private static func simple(_ token: String, in molecule: FMVMolecule) -> [Int] {
        switch token.lowercased() {
        case "all":
            return Array(molecule.atoms.indices)
        case "polymer":
            return molecule.atoms.indices.filter { molecule.atoms[$0].record == "ATOM" }
        case "ligand":
            return molecule.atoms.indices.filter {
                let atom = molecule.atoms[$0]
                return atom.record == "HETATM" && atom.resname.uppercased() != "HOH" && atom.resname.uppercased() != "ZN"
            }
        default:
            return []
        }
    }

    private static func keyed(_ key: String, value: String, in molecule: FMVMolecule) throws -> [Int] {
        switch key.lowercased() {
        case "chain":
            return molecule.atoms.indices.filter { molecule.atoms[$0].chainID.uppercased() == value.uppercased() }
        case "resn":
            return molecule.atoms.indices.filter { molecule.atoms[$0].resname.uppercased() == value.uppercased() }
        case "resi":
            guard let intValue = Int(value) else { throw FMVSelectionError.invalidExpression("invalid resi value: \(value)") }
            return molecule.atoms.indices.filter { molecule.atoms[$0].resseq == intValue }
        case "name":
            return molecule.atoms.indices.filter { molecule.atoms[$0].name.uppercased() == value.uppercased() }
        default:
            throw FMVSelectionError.invalidExpression("unsupported token: \(key)")
        }
    }
}

public struct FlexMolViewPrototype: Sendable {
    public private(set) var molecule: FMVMolecule?

    public init() {}

    public mutating func loadPDB(text: String) {
        self.molecule = FMVParser.parsePDB(text)
    }

    public func atomCount() -> Int {
        molecule?.atomCount ?? 0
    }

    public func select(_ expression: String) throws -> [Int] {
        guard let molecule else { return [] }
        return try FMVSelection.select(expression, in: molecule)
    }
}
