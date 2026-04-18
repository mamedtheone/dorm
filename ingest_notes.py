"""
Dorm-Net: Bulk Note Ingestion Script
Use this to load many course notes at once into the knowledge base.
"""
from brain_module import add_golden_summary

# ─────────────────────────────────────────────────────────────────────────────
# ADD YOUR AASTU COURSE NOTES HERE
# Each entry is a dict with: text, id, and course name
# ─────────────────────────────────────────────────────────────────────────────

NOTES = [
    {
        "id": "calc2_integration_01",
        "course": "Calculus II",
        "text": """
        Calculus II at AASTU focuses heavily on integration techniques:
        1. Integration by Parts: ∫u dv = uv − ∫v du. Choose u to be the term that simplifies when differentiated.
        2. Partial Fractions: Decompose rational functions. Used when denominator has linear/quadratic factors.
        3. Trigonometric Substitution: Replace x with sin, tan, or sec depending on the form under the square root.
        The final exam usually has 3 long theoretical questions and 5 computational problems.
        """
    },
    {
        "id": "physics2_electromagnetism_01",
        "course": "Physics II",
        "text": """
        Physics II at AASTU covers Electromagnetism:
        - Coulomb's Law: F = kq1q2/r². Force between two point charges.
        - Electric Field: E = F/q. Direction is away from positive charges.
        - Gauss's Law: ∮E·dA = Q_enc/ε₀. Useful for symmetric charge distributions.
        - Magnetic Force: F = qv × B. Moving charge in a magnetic field.
        - Faraday's Law: EMF = -dΦ/dt. A changing magnetic field induces an EMF.
        Key tip: Draw diagrams for every problem. Most students lose marks for direction errors.
        """
    },
    {
        "id": "data_structures_01",
        "course": "Data Structures",
        "text": """
        Data Structures midterm at AASTU typically covers:
        - Arrays vs Linked Lists: Arrays have O(1) access but O(n) insertion. Linked lists are reverse.
        - Stacks: LIFO (Last In First Out). Operations: push(), pop(), peek(). Used in function calls.
        - Queues: FIFO (First In First Out). Operations: enqueue(), dequeue(). Used in BFS.
        - Binary Trees: Each node has at most 2 children. Binary Search Tree (BST) rule: left < root < right.
        - Sorting: Bubble sort O(n²), Merge sort O(n log n), Quick sort O(n log n) average.
        """
    },
    {
        "id": "thermodynamics_01",
        "course": "Engineering Thermodynamics",
        "text": """
        Engineering Thermodynamics at AASTU key concepts:
        - First Law: ΔU = Q - W. Energy is conserved. Q is heat added, W is work done by system.
        - Second Law: Entropy of a closed system always increases (or stays the same).
        - Carnot Efficiency: η = 1 - T_cold/T_hot. Maximum possible efficiency of a heat engine.
        - Ideal Gas Law: PV = nRT. Connects pressure, volume, temperature, and moles.
        - Enthalpy: H = U + PV. Useful for constant-pressure processes.
        Exam tip: Always define your system boundary before starting any problem.
        """
    },
    {
        "id": "engineering_math_matrices_01",
        "course": "Engineering Mathematics",
        "text": """
        Engineering Mathematics — Linear Algebra Section:
        - Matrix Multiplication: (AB)ij = sum of row i of A times column j of B.
        - Determinant: For 2x2 [a,b;c,d] = ad - bc. For 3x3, use cofactor expansion.
        - Inverse: A⁻¹ exists only if det(A) ≠ 0. AA⁻¹ = Identity matrix.
        - Eigenvalues: Solve det(A - λI) = 0. Eigenvectors satisfy (A - λI)v = 0.
        - Rank: Number of linearly independent rows/columns. Equal to number of non-zero rows in RREF.
        """
    }
]

def ingest_all():
    print("=== Dorm-Net: Bulk Ingestion Tool ===")
    print(f"Loading {len(NOTES)} course note(s) into the knowledge base...\n")
    
    for note in NOTES:
        add_golden_summary(
            text=note["text"].strip(),
            document_id=note["id"],
            metadata={"course": note["course"]}
        )
    
    print(f"\n[DONE] {len(NOTES)} notes are now in the knowledge base.")
    print("You can now run: python chat.py  -- to start asking questions!")

if __name__ == "__main__":
    ingest_all()
