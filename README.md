# MicroFedML

This project includes implementation of
several secure aggregation protocols,
MicroFedML_1, MicroFedML_2,
[BIK+17](https://eprint.iacr.org/2017/281.pdf),
and [BBG+20](https://eprint.iacr.org/2020/704.pdf),
implemented with [ABIDES](https://github.com/abides-sim/abides) framework.

The full version of [MicroFedML](https://github.com/AnonymousOctupus/MicroFedML/blob/main/MicroFedML.pdf) is also included in the repository.

To run the code,
please first enter the root folder of the project.

To install all required libraries:
<!-- (the file is out of date. Some libraries might need to be installed manually.) -->
```
    pip install -r requirements.txt
```

To run a protocol:
```
    py abides.py -c <config-file-name>
```
The following protocols are runnable:
- `secagg_ro`: the semi-honest version of MicroFedML_1 protocol
- `secagg_group`: the semi-honest version of MicroFedML_2 protocol
- `google_basic`: an implementation of semi-honest version of [BIK+17](https://eprint.iacr.org/2017/281.pdf)
- `google_group`: an implementation of semi-honest version of [BBG+20](https://eprint.iacr.org/2020/704.pdf)
<!-- - `sum`: A basic example of abides framework. -->
<!-- ```
    py abides.py -c google_basic
    py abides.py -c google_group
    py abides.py -c sum
``` -->
For example, to run the MicroFedML_1 protocol with 100 clients and 0.2 offline rate:
```
    py abides.py -c secagg_ro -n 100 --offline_rate 0.2
```

Different protocols have different configurable parameters.
The list of config parameters for a protocol can be found at the beginning of the config file of that protocol in `./config`.
Several important configuration parameters include:
- `-n <integer>`: the total number of users.
- `--input_size <integer>`: the domain of input of each user.
- `--offline_rate <float between 0 and 1>`: the fraction of offline users. **Note:** the default threshold of the protocol is set to 2/3, which means if at least 1/3 users are offline, the protocol will stop without output.
- `-g <integer>`: number of groups, used in protocol MicroFedML_2. It must be at least 2 and `-n` must be divisible by `-g`.
- `--num_neighbors <integer>`: number of neighbors of each user, used in BBG+20.

More example commands can be found in `./experiments` folder
in the corresponding `.sh` files.

The running results can be found
in `./results`.
More specifically,
- `./results/secagg/ro_timing.csv`: running results for MicroFedML_1.
- `./results/secagg/group_timing.csv`: running results for MicroFedML_2.
- `./results/google/basic_timing.csv`: running results for BIK+17.
- `./results/google/group_timing.csv`: running results for BBG+20.

<!-- For more information:
- Section 5 of [this paper](https://github.com/RippleLeaf/abides-secagg/blob/main/docs/Reference.pdf) includes experiments run with abides framework;
- [This document](https://github.com/RippleLeaf/abides-secagg/blob/main/docs/ABIDES_for_Cryptographic_Protocol_Evaluation.pdf)
 contains a brief introduction to the framework.
 It also contains an explanation of the implementation of `sum` protocol. -->
