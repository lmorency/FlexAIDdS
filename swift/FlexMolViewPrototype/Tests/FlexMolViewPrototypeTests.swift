import Testing
@testable import FlexMolViewPrototype

@Test func parsesPDBAndSelectsChain() async throws {
    let pdb = """
ATOM      1  N   ALA A  10       0.000   0.000   0.000  1.00 10.00           N
ATOM      2  CA  ALA A  10       1.000   0.000   0.000  1.00 10.00           C
HETATM    3  C1  LIG B 101       0.000   2.000   0.000  1.00 20.00           C
END
"""
    var view = FlexMolViewPrototype()
    view.loadPDB(text: pdb)
    #expect(view.atomCount() == 3)
    let chainA = try view.select("chain A")
    #expect(chainA.count == 2)
    let ligand = try view.select("ligand")
    #expect(ligand.count == 1)
}
