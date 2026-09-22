#!/usr/bin/env python3
import os
import sys


def insert_kernel_sources(kernel_sources, output_dir, compute_unit):
    compute_unit = compute_unit.strip().rstrip(";")
    ini_path = os.path.join(output_dir, f"aic-{compute_unit}-ops-info.ini")
    if not os.path.exists(ini_path):
        return

    with open(ini_path, encoding="utf-8") as stream:
        lines = stream.readlines()

    for item in kernel_sources.replace(";", ",").split(","):
        fields = item.split()
        if len(fields) < 3:
            continue
        op_type, target_unit, kernel_src = fields[:3]
        if target_unit not in (compute_unit, "ALL"):
            continue

        section = f"[{op_type}]"
        try:
            section_start = lines.index(section + "\n")
        except ValueError:
            continue
        section_end = next(
            (index for index in range(section_start + 1, len(lines))
             if lines[index].strip().startswith("[") and lines[index].strip().endswith("]")),
            len(lines),
        )
        replacement = f"kernelSrc.value={kernel_src}\n"
        for index in range(section_start + 1, section_end):
            if lines[index].strip().startswith("kernelSrc.value="):
                lines[index] = replacement
                break
        else:
            lines.insert(section_start + 1, replacement)

    with open(ini_path, "w", encoding="utf-8") as stream:
        stream.writelines(lines)


if __name__ == "__main__" and len(sys.argv) >= 4:
    insert_kernel_sources(sys.argv[1], sys.argv[2], sys.argv[3])
