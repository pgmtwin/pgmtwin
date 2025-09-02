import os
import argparse
from functools import partial

import numpy as np

from dolfinx import mesh, fem, plot, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import ufl

import pyvista as pv

from common import BasicCantileverSetup


def build_mesh_and_selectors(
    dim_lengths: np.ndarray,
    dim_cells: np.ndarray,
    n_dmg_domains: int,
):
    from dolfinx import mesh
    from mpi4py import MPI

    mesh_cantilever = mesh.create_box(
        MPI.COMM_WORLD,
        [np.zeros_like(dim_lengths), dim_lengths],
        dim_cells.tolist(),
        cell_type=mesh.CellType.hexahedron,
    )

    def bc_clamped_selector(x):
        return np.isclose(x[0], 0)

    def bc_forced_selector(x):
        eps = 1e-5
        aabb_lb = np.array([dim_lengths[0] - dim_lengths[1], 0, dim_lengths[2]])
        aabb_ub = np.array([dim_lengths[0], dim_lengths[1], dim_lengths[2]])

        return np.all(
            (x >= aabb_lb[:, None] - eps) & (x <= aabb_ub[:, None] + eps), axis=0
        )

    def x_selector(x, a, b):
        return (x[0] >= a) & (x[0] <= b)

    x_cuts = np.linspace(0, dim_lengths[0], n_dmg_domains + 1)
    subdomain_selectors = []
    for a, b in zip(x_cuts, x_cuts[1:]):
        subdomain_selectors.append(partial(x_selector, a=a, b=b))

    return (
        mesh_cantilever,
        [bc_clamped_selector, bc_forced_selector],
        subdomain_selectors,
    )


def simulate_hifi(
    output_dir: str,
    damage_location: int = 0,
    youngs_modulus_damage: float = 0.0,
    forcing_pressure: float = 20 * 1e3,
    L: float = 4,
    W: float = 0.3,
    step_length: float = 0.05,
    youngs_modulus: float = 200 * 1e9 / 50,
    poisson_ratio: float = 0.3,
    rho: float = 7801,
    g: float = 9.8,
):
    setup = BasicCantileverSetup()
    damage_location = int(damage_location)

    print(
        f"simulate (hifi) damage location {damage_location} with youngs_modulus_damage {youngs_modulus_damage} and forcing_pressure {forcing_pressure:.2e}"
    )

    domain, [bc_clamped_selector, bc_forced_selector], subdomain_selectors = (
        build_mesh_and_selectors(
            np.array([L, W, W]),
            np.rint([L / step_length, W / step_length, W / step_length]).astype(int),
            len(setup.state_domain.damage_locations) - 1,
        )
    )

    Q = fem.functionspace(domain, ("DG", 0))
    youngs_modulus_field = fem.Function(Q)
    youngs_modulus_field.x.array[:] = youngs_modulus

    if damage_location != setup.state_domain.damage_locations[0]:
        subdomain = mesh.locate_entities(
            domain, domain.topology.dim, subdomain_selectors[damage_location - 1]
        )
        youngs_modulus_field.x.array[subdomain] = np.full_like(
            subdomain,
            youngs_modulus * (1 - youngs_modulus_damage),
            dtype=default_scalar_type,
        )

    lame_lambda = (youngs_modulus_field * poisson_ratio) / (
        (1 + poisson_ratio) * (1 - 2 * poisson_ratio)
    )
    lame_mu = youngs_modulus_field / (2 * (1 + poisson_ratio))

    V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))

    fdim = domain.topology.dim - 1
    bc_clamped_faces = mesh.locate_entities_boundary(domain, fdim, bc_clamped_selector)
    bc_clamped_displ = np.array([0, 0, 0], dtype=default_scalar_type)
    bc_clamped = fem.dirichletbc(
        bc_clamped_displ, fem.locate_dofs_topological(V, fdim, bc_clamped_faces), V
    )

    bc_forced_faces = mesh.locate_entities_boundary(domain, fdim, bc_forced_selector)

    bc_forced_tag = 2
    bc_forced_tags = mesh.meshtags(
        domain,
        fdim,
        bc_forced_faces,
        np.full(len(bc_forced_faces), bc_forced_tag, dtype=np.int32),
    )
    bc_forced_traction = fem.Constant(
        domain, default_scalar_type((0, 0, -forcing_pressure))
    )
    bc_forced_ds = ufl.Measure("ds", domain=domain, subdomain_data=bc_forced_tags)

    f_surface = fem.Constant(domain, default_scalar_type((0, 0, 0)))
    f_volume = fem.Constant(domain, default_scalar_type((0, 0, -rho * g)))

    ds = ufl.Measure("ds", domain=domain)

    def epsilon(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u):
        return lame_lambda * ufl.nabla_div(u) * ufl.Identity(
            len(u)
        ) + 2 * lame_mu * epsilon(u)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    # L = ufl.dot(f_volume, v) * ufl.dx + ufl.dot(f_surface, v) * ds
    L = ufl.dot(f_volume, v) * ufl.dx + ufl.dot(bc_forced_traction, v) * bc_forced_ds

    problem = LinearProblem(
        a, L, bcs=[bc_clamped], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )
    uh = problem.solve()
    strain = epsilon(uh)
    V_eps = fem.functionspace(
        domain,
        (
            "Lagrange",
            1,
            (
                domain.geometry.dim,
                domain.geometry.dim,
            ),
            True,
        ),
    )
    expr_eps = fem.Expression(strain, V.element.interpolation_points())
    eps_h = fem.Function(V_eps)
    eps_h.interpolate(expr_eps)

    os.makedirs(output_dir, exist_ok=True)

    topology, cell_types, geometry = plot.vtk_mesh(V)
    grid = pv.UnstructuredGrid(topology, cell_types, geometry)

    print(f"mesh cells {len(cell_types)}")
    print(f"    nodes {len(geometry)}")

    print(f"displacement size {uh.x.array.size} shape {uh.x.array.shape}")
    print(f"    reshape {(uh.x.array.size / 3, 3)}")
    grid["displacement"] = uh.x.array.reshape((geometry.shape[0], 3))

    print(f"strain size {eps_h.x.array.size} shape {eps_h.x.array.shape}")
    print(f"    reshape {(eps_h.x.array.size / 6, 6)}")
    grid["strain"] = eps_h.x.array.reshape((geometry.shape[0], 6))

    print(
        f"youngs modulus size {youngs_modulus_field.x.array.size} shape {youngs_modulus_field.x.array.shape}"
    )
    print(f"    reshape {(youngs_modulus_field.x.array.size)}")
    grid.cell_data["youngs_modulus"] = youngs_modulus_field.x.array

    fname = os.path.join(output_dir, "results.vtk")
    grid.save(fname, binary=True)

    fname = os.path.join(output_dir, "displacement.npy")
    np.save(fname, grid["displacement"])

    fname = os.path.join(output_dir, "strain.npy")
    np.save(fname, grid["strain"])

    renderer = pv.Plotter(off_screen=True)
    actor_0 = renderer.add_mesh(grid, style="wireframe", color="k")
    warped = grid.warp_by_vector("displacement", factor=1.5)
    actor_1 = renderer.add_mesh(warped, show_edges=True)
    renderer.show_axes()

    fname = os.path.join(output_dir, "displacements.png")
    screenshot = renderer.screenshot(fname)

    renderer.close()

    return uh.x.array


if __name__ == "__main__":
    setup = BasicCantileverSetup()

    # region argparse
    parser = argparse.ArgumentParser(
        description="Simulate a basic cantilever beam, clamped at one end and forced downward at the other"
    )

    parser.add_argument("--db-dir", type=str, required=True, help="output directory")
    parser.add_argument(
        "--dmg-loc",
        type=int,
        default=int(setup.damage_locations[0]),
        required=True,
        help="location of young's modulus reduction",
    )
    parser.add_argument(
        "--youngs-mod-dmg",
        type=float,
        default=setup.digital_damage_levels[0],
        required=True,
        help="reduction of young's modulus",
    )
    parser.add_argument(
        "--pressure",
        type=float,
        default=20000,
        required=True,
        help="value of forcing pressure on free end",
    )

    args = parser.parse_args()
    print(args)
    # endregion

    simulate_hifi(
        database_dir=args.db_dir,
        damage_location=args.dmg_loc,
        youngs_modulus_damage=args.youngs_modu_dmg,
        forcing_pressure=args.pressure,
    )
