from src.generator.generate_dna_sequence import DNASequence
from src.reconstructor.debruijn_reconstructor import DeBruijnReconstructor
from src.reconstructor.rabin_karp_reconstructor import RabinKarpReconstructor
from src.reconstructor.kmp_reconstructor import KMPReconstructor
from src.reconstructor.brute_force_reconstructor import BruteForceReconstructor


def reconstruct_dna(method: str, sequence_file: str, k: int = 31, min_coverage: int = 2):
    reconstructors = {
        'debruijn': DeBruijnReconstructor(k, min_coverage),
        'rabin-karp': RabinKarpReconstructor(k, min_coverage),
        'kmp': KMPReconstructor(k, min_coverage),
        'brute-force': BruteForceReconstructor(k, min_coverage)
    }

    reconstructor = reconstructors[method]
    reconstructed = reconstructor.reconstruct(sequence_file)
    accuracy = reconstructor.validate(sequence_file, reconstructed)
    
    return reconstructed, accuracy 

if __name__ == "__main__":
    generator = DNASequence()
    generator.save_sequence(10000, "test.bin")
    reconstructed, accuracy = reconstruct_dna("brute-force", "test_first10000bp.txt")
    print(f"Reconstructed: {reconstructed}")
    print(f"Accuracy: {accuracy}")
