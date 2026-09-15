"""Compare normalized evaluator JSON, including protocol and per-query traces."""
import argparse
import json
import math
from pathlib import Path


def compare(left, right, *, atol=1e-7, limit=30):
    if not math.isfinite(atol) or atol < 0 or limit < 1:
        raise ValueError("atol must be finite/nonnegative; limit must be positive")
    differences = []
    count = 0

    def difference(path, a, b):
        nonlocal count
        count += 1
        if len(differences) < limit:
            differences.append(dict(path=path, left=a, right=b))

    def visit(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(a.keys() | b.keys()):
                if key not in a or key not in b:
                    difference(path + "." + key, a.get(key, "<missing>"), b.get(key, "<missing>"))
                else:
                    visit(a[key], b[key], path + "." + key)
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                difference(path + ".length", len(a), len(b))
            for i, (av, bv) in enumerate(zip(a, b)):
                visit(av, bv, f"{path}[{i}]")
        elif (isinstance(a, (int, float)) and not isinstance(a, bool) and
              isinstance(b, (int, float)) and not isinstance(b, bool)):
            # Count/rank integers must match exactly; tolerance is for float math.
            equal = a == b if isinstance(a, int) and isinstance(b, int) else abs(a - b) <= atol
            if not equal:
                difference(path, a, b)
        elif type(a) is not type(b) or a != b:
            difference(path, a, b)

    visit(left, right, "$")
    return dict(equal=count == 0, absolute_tolerance=atol,
                difference_count=count, differences=differences, truncated=count > limit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--atol", type=float, default=1e-7)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    def reject_constant(value):
        raise ValueError("non-standard JSON number: " + value)

    result = compare(json.loads(args.left.read_text(), parse_constant=reject_constant),
                     json.loads(args.right.read_text(), parse_constant=reject_constant),
                     atol=args.atol, limit=args.limit)
    serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(serialized)
    print(serialized, end="")
    return 0 if result["equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
