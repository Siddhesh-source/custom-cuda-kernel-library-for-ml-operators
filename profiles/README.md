# profiles

Small Nsight Compute (`.ncu-rep`) and Nsight Systems (`.nsys-rep`) reports, one per major optimization (reference.md §7, §9). Layout: `profiles/<op>/<algo>_<shape>_<arch>.ncu-rep`. Keep each file small: use `--kernel-name` and `--launch-count 1`.
