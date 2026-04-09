from perception.objects import Object, extract_objects
from perception.colors import ColorAnalysis, analyze_colors
from perception.symmetry import SymmetryResult, analyze_symmetry
from perception.patterns import PeriodicityResult, detect_periodicity
from perception.topology import ObjectRelation, relate, relate_all
from perception.lines import Line, find_lines
from perception.report import GridPerception, PairPerception, PerceptionReport, perceive
