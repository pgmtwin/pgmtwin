"""
Utilities for processing dependencies and building Dynamic Bayesian Networks from them.
"""

import copy
import itertools as itools
from typing import Dict, List, Optional, Tuple

import numpy as np
import networkx as nx

import joblib

from pgmpy.models import DynamicBayesianNetwork
from pgmpy.inference import DBNInference

from pgmpy.factors.discrete import TabularCPD

from pgmtwin.core.utils import (
    filter_kwargs,
    pgmpy_suppress_cpd_replacement_warning,
    sample_joint_distributions,
)

from .variable import Variable
from .dependency import Dependency


def compress_dependencies(
    dependencies: List[Dependency], cpd_smoothing: float = 0, verbose: bool = False
) -> List[Dependency]:
    """
    Process the dependencies to generate a new set where:
    - every output has a single dependency grouping all inputs
    - every node is present in all frames

    Args:
        dependencies (List[Dependency]): the list of dependencies to process
        cpd_smoothing (bool, optional): smoothing when normalizing combined cpds. Defaults to 0.
        verbose (bool, optional): whether to print debug messages. Defaults to False.

    Returns:
        List[Dependency]: the processed list of dependencies
    """
    # group deps by output, get names and frames
    deps_by_output: Dict[Variable, List[Dependency]] = {}
    vnames = set()
    frames = set()
    for d in dependencies:
        if d.output not in deps_by_output:
            deps_by_output[d.output] = [d]
        else:
            deps_by_output[d.output].append(d)

        vnames.add(d.output[0])
        frames.add(d.output[1])
        for n, f in d.inputs:
            vnames.add(n)
            frames.add(f)
    frames = sorted(list(frames))

    deps_by_output = dict(sorted(deps_by_output.items()))

    if verbose:
        print("initial dependencies grouped by output")
        for o, ds in deps_by_output.items():
            print(f"{o}:")
            for d in ds:
                print(f"    {d}")
        print("")

    # fill the missing (var, frame) spots
    missing = set(itools.product(vnames, frames[1:])).difference(deps_by_output)
    missing = [m for m in missing if (m[0], m[1] - 1) in deps_by_output]
    while missing:
        if missing and verbose:
            print(f"missing frame vars {missing}")

        for m_var, m_f in missing:
            if verbose:
                print(f"    adding {(m_var, m_f)}")
            deps_by_output[(m_var, m_f)] = [
                d.get_timestep_copy(1) for d in deps_by_output[(m_var, m_f - 1)]
            ]

        if not missing:
            break

        missing = set(itools.product(vnames, frames[1:])).difference(deps_by_output)
        missing = [m for m in missing if (m[0], m[1] - 1) in deps_by_output]
    else:
        if verbose:
            deps_by_output = dict(sorted(deps_by_output.items()))

            print("after adding missing frame vars")
            for o, ds in deps_by_output.items():
                print(f"{o}:")
                for d in ds:
                    print(f"    {d}")
            print("")

    # sort the dict for timestep duplication loop
    deps_by_output = dict(sorted(deps_by_output.items()))

    for d in dependencies:
        o_var, o_f = d.output
        for f in frames:
            if o_f >= f:
                continue

            # check if output is in dict
            k = (o_var, f)
            if k in deps_by_output:
                # if present, check if current dep can be added to deps list
                if all(d.name != d2.name for d2 in deps_by_output[k]):
                    if verbose:
                        print(f"adding {d.name} to {k}")
                    deps_by_output[k].append(d.get_timestep_copy(f - o_f))
    # re-sort
    deps_by_output = dict(sorted(deps_by_output.items()))

    if missing and verbose:
        print("after dependencies timestep duplication")
        for o, ds in deps_by_output.items():
            print(f"{o}:")
            for d in ds:
                print(f"    {d}")

    # for each output, merge the cpds from the parents
    for o in deps_by_output:
        ds = deps_by_output[o]
        d = ds[0]

        if len(ds) > 1:
            for d2 in ds[1:]:
                d = d.merge(d2, cpd_smoothing=cpd_smoothing)

        deps_by_output[o] = d

    # re-sort
    deps_by_output = dict(sorted(deps_by_output.items()))
    if verbose:
        print(f"simplified dependencies list")
        for o, ds in deps_by_output.items():
            print(f"{o}:")
            print(f"    {ds}")
        print("")

    return list(deps_by_output.values())


def get_filtered_discrete_metadata(
    variables: List[Variable],
    dependencies: List[Dependency],
) -> Tuple[List[Variable], List[Dependency]]:
    """
    Retrieve new lists where non-discrete variables, and dependencies touching them, have been removed

    Args:
        variables (List[Variable]): the list of variables to filter
        dependencies (List[Dependency]): the list of dependencies to filter

    Returns:
        Tuple[List[Variable], List[Dependency]]: the filtered variables and dependencies
    """
    v2 = [v for v in variables if v.discrete]
    d2 = [
        d
        for d in dependencies
        if d.output[0] in v2 and all(i in v2 for i, _ in d.inputs)
    ]

    vs = set()
    for d in d2:
        vs.add(d.output[0])
        vs.update([i for i, _ in d.inputs])

    assert vs.issubset(
        set(v2)
    ), f"dependencies use unknown variables {vs.difference(v2)}"

    return v2, d2


def get_dbn(
    dependencies: List[Dependency],
    check_model: bool = True,
    cpd_smoothing: float = 0,
    verbose: bool = False,
) -> Tuple[DynamicBayesianNetwork, List[Tuple[Variable, int]]]:
    """
    Builds a DynamicBayesianNetwork from the given dependencies
    The dependencies are first compressed, and the inputs, output relations become edges
    The cpds are applied where available, and source nodes are given a priori uniform distributions

    Args:
        dependencies (List[Dependency]): the list of dependencies to process
        check_model (bool, optional): check the dbn before returning it. Defaults to True
        cpd_smoothing (bool, optional): smoothing when normalizing combined cpds. Defaults to 0.
        verbose (bool, optional): whether to print debug messages. Defaults to False.

    Returns:
        Tuple[DynamicBayesianNetwork, List[Tuple[Variable, int]]]: the constructed DynamicBayesianNetwork and the source nodes found
    """
    dependencies = compress_dependencies(
        dependencies, cpd_smoothing=cpd_smoothing, verbose=verbose
    )

    source_nodes = set()
    for d in dependencies:
        source_nodes.update(d.inputs)
        source_nodes.add(d.output)

    for d in dependencies:
        source_nodes.remove(d.output)

    source_nodes = sorted(list(source_nodes))

    if verbose:
        print(f"source nodes:")
        for s in source_nodes:
            print(f"    {s}")
        print("")

    ret = DynamicBayesianNetwork()

    dbn_edges = []
    dbn_cpds: List[TabularCPD] = []

    # dependencies give edges and cpds
    for d in dependencies:
        out_var, out_f = d.output
        in_vars = [v for v, _ in d.inputs]
        in_fs = [f for _, f in d.inputs]
        for p_var, p_f in d.inputs:
            dbn_edges.append(((p_var.name, p_f), (out_var.name, out_f)))

        if d.cpd is not None:
            if verbose:
                print(f"adding {d} as cpd")
                print(f"    len(out_var.domain) {len(out_var.domain)}")
                print(f"    d.cpd {d.cpd.shape}")
                print(f"    evidence {[(v.name, f) for v, f in zip(in_vars, in_fs)]}")
                print(f"    evidence_card {[len(v.domain) for v in in_vars]}")

            cpd = TabularCPD(
                (out_var.name, out_f),
                len(out_var.domain),
                values=d.cpd,
                evidence=[(v.name, f) for v, f in zip(in_vars, in_fs)],
                evidence_card=[len(v.domain) for v in in_vars],
            )
            dbn_cpds.append(cpd)

    ret.add_edges_from(dbn_edges)

    # source nodes will get an a priori cpd, a uniform distribution
    for out_var, out_f in source_nodes:
        if verbose:
            print(f"adding a priori cpd to {(out_var, out_f)}")
            print(f"    len(out_var.domain) {len(out_var.domain)}")
            print(f"    p(elem) {1 / len(out_var.domain)}")

        cpd = TabularCPD(
            variable=(out_var.name, out_f),
            variable_card=len(out_var.domain),
            values=np.ones(len(out_var.domain)).reshape(-1, 1),
        )
        dbn_cpds.append(cpd)

    for t in dbn_cpds:
        t.normalize()

    with pgmpy_suppress_cpd_replacement_warning():
        ret.add_cpds(*dbn_cpds)

    if check_model:
        ret.check_model()

    return ret, source_nodes


def draw_structured(
    dbn: DynamicBayesianNetwork,
    var_pos: Dict[str, Tuple[float, float]],
    var_label: Dict[str, str] = None,
    var_color: Dict[str, str] = None,
    var_shape: Dict[str, str] = None,
    frame_grid_step: Tuple[float, float] = (5, 3),
    **kwargs,
):
    """
    Draws the given DynamicBayesianNetwork, controlling the placement on nodes based on their str name

    Args:
        dbn (DynamicBayesianNetwork): the DynamicBayesianNetwork to draw
        var_pos (Dict[str, Tuple[float, float]]): specifies the position of nodes, relative to a frame. The keys are the str part of nodes
        var_label (Dict[str, str], optional): if given, converts the string part of the nodes to the specified value. Frames are rendered in square brackets. Defaults to None.
        var_color (Dict[str, str], optional): if given, specifies the matplotlib color to paint the nodes of the corresponding variable. Defaults to None.
        var_shape (Dict[str, str], optional): if given, specifies the matplotlib shape to draw the nodes of the corresponding variable. Frames are rendered in square brackets. Defaults to None.
        frame_grid_step (Tuple[float, float], optional): the horizontal and vertical spacing between coordinates in a frame. Defaults to (5, 3).
    """
    frame_size = max(x for x, _ in var_pos.values())

    node_pos = {}
    node_labels = {}
    node_color = {}
    node_shape = {}
    node_shape_selectors: Dict[str, List[int]] = {}
    for i, node in enumerate(dbn.nodes):
        (n, f) = node
        x, y = var_pos[n]

        node_pos[node] = (
            ((frame_size + 2) * f + x) * frame_grid_step[0],
            y * frame_grid_step[1],
        )

        if var_label and n in var_label:
            node_labels[node] = var_label[n] + f"[{f}]"
        else:
            node_labels[node] = str(node)

        if var_color and n in var_color:
            node_color[node] = var_color[n]
        else:
            node_color[node] = "#1f78b4"

        if var_shape and n in var_shape:
            node_shape[node] = var_shape[n]
            s = var_shape[n]
        else:
            node_shape[node] = "o"
            s = "o"

        s = node_shape[node]
        if s in node_shape_selectors:
            node_shape_selectors[s].append(i)
        else:
            node_shape_selectors[s] = [i]

        assert node in node_labels

    node_color = np.array(list(node_color.values()))

    # kwargs["ax"] = ax

    if len(node_shape_selectors) <= 1:
        nx.draw_networkx(
            dbn,
            pos=node_pos,
            labels=node_labels,
            node_color=node_color,
            node_shape=next(iter(node_shape_selectors)),
            **kwargs,
        )
    else:
        # first draw the nodes of each shape
        nodes_kwargs = filter_kwargs(nx.draw_networkx_nodes, kwargs)
        for s, selector in node_shape_selectors.items():
            selector = np.array(selector)
            nodelist = list(n for i, n in enumerate(dbn.nodes) if i in selector)

            nx.draw_networkx_nodes(
                dbn,
                nodelist=nodelist,
                pos=node_pos,
                node_color=node_color[selector],
                node_shape=s,
                **nodes_kwargs,
            )

        # then the rest
        edges_kwargs = filter_kwargs(nx.draw_networkx_edges, kwargs)
        nx.draw_networkx_edges(
            dbn,
            pos=node_pos,
            **edges_kwargs,
        )
        labels_kwargs = filter_kwargs(nx.draw_networkx_labels, kwargs)
        nx.draw_networkx_labels(
            dbn,
            pos=node_pos,
            labels=node_labels,
            **labels_kwargs,
        )


def dbn_inference_worker(
    dbn_infer: DBNInference,
    inference_keys: List[Tuple[str, int]],
    evidence: Dict[Tuple[str, int], np.ndarray],
    evidence_weights: Optional[np.ndarray] = None,
) -> Dict[Tuple[str, int], np.ndarray]:
    """
    Performs a batch of inference experiments via sampling

    Args:
        dbn_infer (DBNInference): the dbn to infer from
        inference_keys (List[Tuple[str, int]]): the list of output variables
        evidence (Dict[Tuple[str, int], np.ndarray]): the sampled evidence for the evidence nodes
        evidence_weights (np.ndarray, optional): the weighting vector of the evidence samples, or None to use uniform weighting. Defaults to None.

    Raises:
        ValueError: if no samples were given for an evidence node

    Returns:
        Dict[Tuple[str, int], np.ndarray]: the inferred sample count arrays
    """
    n_max = 0
    for k, vs in evidence.items():
        n = len(vs)
        assert n > 0, f"samples for evidence {k} must be at least 1, got {n}"

        n_max = max(n_max, n)

    if evidence_weights is None:
        evidence_weights = np.full(n_max, 1)
    else:
        assert len(evidence_weights) == n_max

    local_evidence = {}
    local_inference = {}
    for i, w in enumerate(evidence_weights):
        for k, vs in evidence.items():
            v = None
            if isinstance(vs, np.ndarray):
                if vs.size == 1:
                    v = int(vs[0])
                else:
                    v = int(vs[i])
            else:
                raise ValueError(
                    f"evidence samples must be given as list or np.ndarray"
                )
            local_evidence[k] = v

        with pgmpy_suppress_cpd_replacement_warning():
            inference = dbn_infer.forward_inference(inference_keys, local_evidence)

        for k in inference_keys:
            if k in local_inference:
                local_inference[k] += inference[k].values * w
            else:
                local_inference[k] = inference[k].values * w

    return local_inference


def dbn_inference(
    dbn_infer: DBNInference,
    inference_keys: List[Tuple[str, int]],
    evidence: Dict[Tuple[str, int], int],
    n_samples: int,
    evidence_sampled_distros: Optional[Dict[Tuple[str, int], np.ndarray]] = None,
    rng: Optional[np.random.Generator] = None,
    n_workers: Optional[int] = 1,
    use_weighting: bool = True,
    verbose: bool = False,
) -> Dict[Tuple[str, int], np.ndarray]:
    """
    Performs inference experiments on the given dbn, via sampling of the evidence nodes

    Args:
        dbn_infer (DBNInference): the dbn to infer from
        inference_keys (List[Tuple[str, int]]): the list of output variables
        evidence (Dict[Tuple[str, int], int]): hard evidence, these nodes will be fixed and not sampled
        n_samples (int): number of samples to draw from the product of evidence_sampled_distros
        evidence_sampled_distros (Optional[Dict[Tuple[str, int], np.ndarray]], optional): pmf of each evidence node to be sampled. Defaults to None.
        rng (Optional[np.random.Generator], optional): pseudo-random generator, will create a new one if None. Defaults to None.
        n_workers (Optional[int], optional): number of parallel batches to split the workload in, or joblib.cpu_count(only_physical_cores=True) if None . Defaults to None.
        use_weighting (bool, optional): whether to sample considering the evidence distributions as weights. Defaults to True.
        verbose (bool, optional): whether to print debug messages. Defaults to False.

    Raises:
        NotImplementedError: if the distributions are not given as np.ndarray

    Returns:
        Dict[Tuple[str, int], np.ndarray]: the inferred distributions
    """
    if rng is None:
        rng = np.random.default_rng()

    evidence = copy.deepcopy(evidence)
    # inference is deterministic, so sampling should exploit that
    if use_weighting:
        pmfs = list(evidence_sampled_distros.values())
        sample_idxs, evidence_weights = sample_joint_distributions(
            n_samples, pmfs, rng=rng
        )

        if verbose:
            print(f"preparing sample weights")
            for k, distro in evidence_sampled_distros.items():
                print(f"evidence {k}")
                print(f"    {distro}")
                print(f"    has {np.sum(~np.isclose(distro, 0))} nonzeroes")
            print(
                f"requested n_samples {n_samples} generated len(sample_idxs) {len(sample_idxs)}"
            )

            print(f"samples")
            for idxs, w in zip(sample_idxs, evidence_weights):
                print(idxs, w)

        n_samples = len(sample_idxs)
        assert n_samples, f"found 0 samples or invalid distributions"

        sample_idxs = np.array(sample_idxs).T
        for k, samples in zip(evidence_sampled_distros, sample_idxs):
            evidence[k] = samples
    else:
        evidence_weights = None

        for k, distro in evidence_sampled_distros.items():
            evidence[k] = rng.choice(
                len(distro),
                size=n_samples,
                p=distro,
            )

    if n_workers is None:
        n_workers = joblib.cpu_count(only_physical_cores=True)

    n_workers = min(n_samples, n_workers)
    assert n_workers, f"found 0 workers"

    for k, v in evidence.items():
        if isinstance(v, np.ndarray):
            evidence[k] = np.atleast_1d(v)
        else:
            evidence[k] = np.atleast_1d([v])

    if n_workers == 1:
        ret = dbn_inference_worker(
            dbn_infer,
            inference_keys=inference_keys,
            evidence=evidence,
            evidence_weights=evidence_weights,
        )
    else:
        chunks = []

        batch = n_samples // n_workers
        i_start = 0
        for j in range(n_workers):
            i_size = batch + (j < n_samples % batch)

            e = {}
            for k, vs in evidence.items():
                if isinstance(vs, np.ndarray):
                    if vs.size == 1:
                        v = np.atleast_1d(vs[0:1])
                    else:
                        v = np.atleast_1d(vs[i_start : i_start + i_size])
                else:
                    raise NotImplementedError("only arrays are supported")
                e[k] = v
            w = None
            if use_weighting:
                w = evidence_weights[i_start : i_start + i_size]

            chunks.append((e, w))

            i_start += i_size

        results: List[Dict[(str, int), np.ndarray]] = joblib.Parallel(n_jobs=n_workers)(
            joblib.delayed(dbn_inference_worker)(dbn_infer, inference_keys, e, w)
            for (e, w) in chunks
        )

        ret: Dict[(str, int), np.ndarray] = results[0]
        for e in results[1:]:
            for k in ret:
                ret[k] += e[k]

    for k in ret:
        ret[k] /= np.sum(ret[k])

    return ret
