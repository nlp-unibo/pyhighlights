"""Ruggeri and Signorelli, 2025, *Interlocking-free Selective Rationalization
Through Genetic-based Learning*, ACL 2025.

Paper: <https://aclanthology.org/2025.acl-long.59/>.
Reference implementation: <https://github.com/nlp-unibo/gen-spp>.

Two corpora -- a synthetic one and HateXplain -- against FR, MGR, MCD, G-RAT
and GenSPP. Every value here is the released implementation's; where the paper
and the release disagree, or where pyhighlights cannot reproduce something
exactly, the configuration says so at the point it matters.
"""

from pyhighlights_benchmarks.genspp2025.corpora import GenSPPToyLoader

__all__ = ["GenSPPToyLoader"]
