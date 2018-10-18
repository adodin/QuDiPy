import unittest
from unittest import TestCase
import QuDiPy.containers.grids as gr
import QuDiPy.coordinates.spherical as sp
import numpy as np
from numpy import pi
import QuDiPy.math.linear_algebra as la


'''class TestVectorizeOperatorGrid(TestCase):
    def test_vectorize_cartesian(self):
        si = la.CartesianSpinOperator((1, 0, 0, 0))
        sx = la.CartesianSpinOperator((0, 1, 0, 0))
        sy = la.CartesianSpinOperator((0, 0, 1, 0))
        sz = la.CartesianSpinOperator((0, 0, 0, 1))
        ops = np.array([[si, sx], [sy, sz]])
        expected = (np.array([[1, 0], [0, 0]]), np.array([[0, 1], [0, 0]]),
                    np.array([[0, 0], [1, 0]]), np.array([[0, 0], [0, 1]]))

        x1 = (0, 1)
        x2 = (0, 1)
        grid = gr.CartesianGrid((x1, x2))

        op_grid = gr.GridData(ops, grid)
        expected = gr.GridData(expected, grid)
        calculated = gr.vectorize_operator_grid(op_grid)

        self.assertEqual(expected, calculated)

    def test_vectorize_spherical(self):
        si = la.SphericalSpinOperator((1, 0, 0, 0))
        sx = la.SphericalSpinOperator((0, 1, pi/2, 0))
        sy = la.SphericalSpinOperator((0, 1, pi/2, pi/2))
        sz = la.SphericalSpinOperator((0, 1, 0, 0))
        ops = np.array([[si, sx], [sy, sz]])
        expected = (np.array([[1, 0], [0, 0]]), np.array([[0, 1], [0, 0]]),
                    np.array([[0, 0], [1, 0]]), np.array([[0, 0], [0, 1]]))

        x1 = (0, 1)
        x2 = (0, 1)
        grid = gr.CartesianGrid((x1, x2))

        op_grid = gr.GridData(ops, grid)
        expected = gr.GridData(expected, grid)
        calculated = gr.vectorize_operator_grid(op_grid)

        self.assertAlmostEqual(expected, calculated)


class TestUnvectorizeOperatorGrid(TestCase):
    def test_unvectorize_cartesian(self):
        si = la.CartesianSpinOperator((1, 0, 0, 0))
        sx = la.CartesianSpinOperator((0, 1, 0, 0))
        sy = la.CartesianSpinOperator((0, 0, 1, 0))
        sz = la.CartesianSpinOperator((0, 0, 0, 1))
        ops = np.array([[si, sx], [sy, sz]])

        x1 = (0, 1)
        x2 = (0, 1)
        grid = gr.CartesianGrid((x1, x2))

        op_grid = gr.GridData(ops, grid)
        expected = op_grid
        calculated = gr.vectorize_operator_grid(op_grid)
        calculated = gr.unvectorize_operator_grid(calculated, la.CartesianSpinOperator)

        self.assertEqual(expected, calculated)


class TestInitializeOperatorGrid(TestCase):
    def test_initialization(self):
        x = (0., 1.)
        y = (0., 1.)
        z = (0., 1.)
        grid = gr.CartesianGrid((x, y, z))
        s000 = la.CartesianSpinOperator((0., 0., 0., 0.))
        s001 = la.CartesianSpinOperator((0., 0., 0., 1.))
        s010 = la.CartesianSpinOperator((0., 0., 1., 0.))
        s011 = la.CartesianSpinOperator((0., 0., 1., 1.))
        s100 = la.CartesianSpinOperator((0., 1., 0., 0.))
        s101 = la.CartesianSpinOperator((0., 1., 0., 1.))
        s110 = la.CartesianSpinOperator((0., 1., 1., 0.))
        s111 = la.CartesianSpinOperator((0., 1., 1., 1.))
        ops = np.array([[[s000, s001], [s010, s011]], [[s100, s101], [s110, s111]]])
        o_grid_ex = gr.GridData(ops, grid)
        o_grid_ac = gr.initialize_operator_grid(grid, la.CartesianSpinOperator, i_coord=0.)
        self.assertEqual(o_grid_ac, o_grid_ex)
'''


# Initialize Cartesian Coordinate Arrays
x = np.linspace(-1., 1., 21)
y = np.linspace(-1., 1., 21)
z = np.linspace(-1., 1., 21)
c_grid = gr.CartesianGrid((x, y, z))
x_short = np.linspace(-1., 1., 11)
y_short = np.linspace(-1., 1., 11)
z_short = np.linspace(-1., 1., 11)
short_c_grid = gr.CartesianGrid((x_short, y_short, z_short))

# Initialize Spherical Coordinate Arrays
r = np.linspace(0., 1., 21)
t = pi * np.linspace(0., 1, 21)
p = 2 * pi * np.linspace(0., 1, 21)
s_grid = gr.SphericalGrid((r, t, p))
r_short = np.linspace(0., 1., 11)
t_short = pi * np.linspace(0., 1, 11)
p_short = 2 * pi * np.linspace(0., 1, 11)
short_s_grid = gr.SphericalGrid((r_short, t_short, p_short))
r_log = np.logspace(-3, 0, 11)
t_log = pi * np.logspace(-3, 0, 11)
p_log = 2 * pi * np.logspace(-5, 0, 11)


# Test Grid Classes
class TestCartesianGrid(TestCase):
    def test_cartesian_grid_init_equal_length_inputs_mesh_equality(self):
        # Initialize Grids with equal length coordinate arrays
        short_grid = gr.CartesianGrid((x_short, y_short, z_short))
        long_grid = gr.CartesianGrid((x, y, z))

        # Define Expected Behavior
        expected_short_mesh = np.meshgrid(x_short, y_short, z_short, indexing='ij')
        expected_long_mesh = np.meshgrid(x, y, z, indexing='ij')

        # Check Equality
        short_mesh_eq = np.array_equal(short_grid.mesh, expected_short_mesh)
        long_mesh_eq = np.array_equal(long_grid.mesh, expected_long_mesh)
        self.assertTrue(short_mesh_eq)
        self.assertTrue(long_mesh_eq)

    def test_cartesian_grid_init_equal_length_inputs_cartesian_equality(self):
        # Initialize Grids with equal length coordinate arrays
        short_grid = gr.CartesianGrid((x_short, y_short, z_short))
        long_grid = gr.CartesianGrid((x, y, z))

        # Define Expected Behavior
        expected_short_cart = np.meshgrid(x_short, y_short, z_short, indexing='ij')
        expected_long_cart = np.meshgrid(x, y, z, indexing='ij')

        # Check Equality
        short_mesh_eq = np.array_equal(short_grid.cartesian, expected_short_cart)
        long_mesh_eq = np.array_equal(long_grid.cartesian, expected_long_cart)
        self.assertTrue(short_mesh_eq)
        self.assertTrue(long_mesh_eq)

    def test_cartesian_grid_init_unequal_length_inputs_mesh_equality(self):
        # Initialize Grids with equal length coordinate arrays
        mixed_grid = gr.CartesianGrid((x, y_short, z_short))

        # Define Expected Behavior
        expected_mixed_mesh = np.meshgrid(x, y_short, z_short, indexing='ij')

        # Check Equality
        mixed_mesh_eq = np.array_equal(mixed_grid.mesh, expected_mixed_mesh)
        self.assertTrue(mixed_mesh_eq)

    def test_cartesian_grid_init_unequal_length_inputs_cartesian_equality(self):
        # Initialize Grids with equal length coordinate arrays
        mixed_grid = gr.CartesianGrid((x, y_short, z_short))

        # Define Expected Behavior
        expected_mixed_cart = np.meshgrid(x, y_short, z_short, indexing='ij')

        # Check Equality
        mixed_cart_eq = np.array_equal(mixed_grid.cartesian, expected_mixed_cart)
        self.assertTrue(mixed_cart_eq)

    def test_cartesian_grid_init_list_implicit_container_type(self):
        grid = gr.CartesianGrid([x_short, y_short, z_short])
        container_type = grid._container_type
        coordinate_type = type(grid.coordinates)
        mesh_type = type(grid.mesh)
        cartesian_type = type(grid.cartesian)
        self.assertIs(container_type, list)
        self.assertIs(coordinate_type, list)
        self.assertIs(mesh_type, list)
        self.assertIs(cartesian_type, list)

    def test_cartesian_grid_eq_equal_length_grids_return_true(self):
        x2 = 1. * x       # Use this so that x2 == x but is not x
        assert x2 is not x
        assert np.array_equal(x, x2)
        grid = gr.CartesianGrid((x, y, z))
        grid2 = gr.CartesianGrid((x2, y, z))
        self.assertEqual(grid, grid2)

    def test_cartesian_grid_eq_very_different_equal_length_grids_return_false(self):
        x2 = 2. * x
        assert not np.array_equal(x, x2)
        grid = gr.CartesianGrid((x, y, z))
        grid2 = gr.CartesianGrid((x2, y, z))
        self.assertNotEqual(grid, grid2)

    def test_cartesian_grid_eq_nearly_equal_length_grids_return_false(self):
        x2 = x +1E-7 * np.ones_like(x)
        assert x2 is not x
        assert not np.array_equal(x, x2)
        grid = gr.CartesianGrid((x, y, z))
        grid2 = gr.CartesianGrid((x2, y, z))
        self.assertNotEqual(grid, grid2)

    def test_cartesian_grid_getitem_equal_length_arrays(self):
        grid = gr.CartesianGrid((x, y, z))
        expected_000 = (x[0], y[0], z[0])
        expected_210 = (x[2], y[1], z[0])
        self.assertEqual(grid[0, 0, 0], expected_000)
        self.assertEqual(grid[2, 1, 0], expected_210)

    def test_cartesian_grid_getitem_unequal_length_arrays(self):
        grid = gr.CartesianGrid((x_short, y, z))
        expected_000 = (x_short[0], y[0], z[0])
        expected_210 = (x_short[2], y[1], z[0])
        self.assertEqual(grid[0, 0, 0], expected_000)
        self.assertEqual(grid[2, 1, 0], expected_210)

    def test_cartesian_grid_getitem_type_list(self):
        grid = gr.CartesianGrid([x, y, z])
        getitem_type = type(grid[0, 0, 0])
        self.assertIs(getitem_type, list)
        self.assertIs(getitem_type, grid._container_type)

    def test_cartesian_grid_mesh_item_equal_length_arrays(self):
        grid = gr.CartesianGrid((x, y, z))
        expected_000 = (x[0], y[0], z[0])
        expected_210 = (x[2], y[1], z[0])
        self.assertEqual(grid.mesh_item(0, 0, 0), expected_000)
        self.assertEqual(grid.mesh_item(2, 1, 0), expected_210)

    def test_cartesian_grid_mesh_item_unequal_length_arrays(self):
        grid = gr.CartesianGrid((x_short, y, z))
        expected_000 = (x_short[0], y[0], z[0])
        expected_210 = (x_short[2], y[1], z[0])
        self.assertEqual(grid.mesh_item(0, 0, 0), expected_000)
        self.assertEqual(grid.mesh_item(2, 1, 0), expected_210)

    def test_cartesian_grid_mesh_item_type_list(self):
        grid = gr.CartesianGrid([x, y, z])
        getitem_type = type(grid.mesh_item(0, 0, 0))
        self.assertIs(getitem_type, list)
        self.assertIs(getitem_type, grid._container_type)

    def test_cartesian_grid_iter_tuples(self):
        grid = gr.CartesianGrid((x, y, z))
        xm, ym, zm = np.meshgrid(x, y, z, indexing='ij')
        x_expected = xm.flatten()
        y_expected = ym.flatten()
        z_expected = zm.flatten()
        for c, xe, ye, ze in zip(grid, x_expected, y_expected, z_expected):
            self.assertEqual(c, (xe, ye, ze))

    def test_cartesian_grid_iter_lists(self):
        grid = gr.CartesianGrid([x, y, z])
        xm, ym, zm = np.meshgrid(x, y, z, indexing='ij')
        x_expected = xm.flatten()
        y_expected = ym.flatten()
        z_expected = zm.flatten()
        for c, xe, ye, ze in zip(grid, x_expected, y_expected, z_expected):
            self.assertEqual(c, [xe, ye, ze])

    def test_cartesian_grid_iter_types(self):
        tuple_grid = gr.CartesianGrid((x, y, z))
        list_grid = gr.CartesianGrid([x, y, z])
        for t, l in zip(tuple_grid, list_grid):
            self.assertIs(type(t), tuple)
            self.assertIs(type(l), list)

    def test_cartesian_grid_mesh_iter_tuples(self):
        grid = gr.CartesianGrid((x, y, z))
        xm, ym, zm = np.meshgrid(x, y, z, indexing='ij')
        x_expected = xm.flatten()
        y_expected = ym.flatten()
        z_expected = zm.flatten()
        for c, xe, ye, ze in zip(grid.mesh_iter(), x_expected, y_expected, z_expected):
            self.assertEqual(c, (xe, ye, ze))

    def test_cartesian_grid_mesh_iter_lists(self):
        grid = gr.CartesianGrid([x, y, z])
        xm, ym, zm = np.meshgrid(x, y, z, indexing='ij')
        x_expected = xm.flatten()
        y_expected = ym.flatten()
        z_expected = zm.flatten()
        for c, xe, ye, ze in zip(grid.mesh_iter(), x_expected, y_expected, z_expected):
            self.assertEqual(c, [xe, ye, ze])

    def test_cartesian_grid_mesh_iter_types(self):
        tuple_grid = gr.CartesianGrid((x, y, z))
        list_grid = gr.CartesianGrid([x, y, z])
        for t, l in zip(tuple_grid.mesh_iter(), list_grid.mesh_iter()):
            self.assertIs(type(t), tuple)
            self.assertIs(type(l), list)

    def test_cartesian_grid_volume_equal_length_arrays(self):
        dx, dy, dz = 0.1 * np.ones_like(x), 0.1 * np.ones_like(y), 0.1 * np.ones_like(z)
        dx[0] = 0.5 * dx[0]
        dx[-1] = 0.5 * dx[1]
        dy[0] = 0.5 * dy[0]
        dy[-1] = 0.5 * dy[-1]
        dz[0] = 0.5 * dz[0]
        dz[-1] = 0.5 * dz[-1]
        dxg, dyg, dzg = np.meshgrid(dx, dy, dz, indexing='ij')
        v_e = dxg * dyg * dzg

        # Round For float equality check
        grid = gr.CartesianGrid((x, y, z))
        v_e = np.round(v_e, 7)
        v_c = np.round(grid.volume, 7)
        self.assertTrue(np.array_equal(v_e, v_c))

    def test_cartesian_grid_volume_unequal_length_arrays(self):
        dx, dy, dz = 0.2 * np.ones_like(x_short), 0.1 * np.ones_like(y), 0.1 * np.ones_like(z)
        dx[0] = 0.5 * dx[0]
        dx[-1] = 0.5 * dx[1]
        dy[0] = 0.5 * dy[0]
        dy[-1] = 0.5 * dy[-1]
        dz[0] = 0.5 * dz[0]
        dz[-1] = 0.5 * dz[-1]
        dxg, dyg, dzg = np.meshgrid(dx, dy, dz, indexing='ij')
        v_e = dxg * dyg * dzg

        # Round For float equality check
        grid = gr.CartesianGrid((x_short, y, z))
        v_e = np.round(v_e, 7)
        v_c = np.round(grid.volume, 7)
        self.assertTrue(np.array_equal(v_e, v_c))

    def test_cartesian_grid_gradient_equal_length_arrays_linear(self):
        xf, yf, zf = np.meshgrid(x, y, z, indexing='ij')
        grid = gr.CartesianGrid((x, y, z))
        grad_x = grid.gradient(xf)
        grad_y = grid.gradient(yf)
        grad_z = grid.gradient(zf)

        grad_xe = (np.ones_like(xf), np.zeros_like(yf), np.zeros_like(zf))
        grad_ye = (np.zeros_like(xf), np.ones_like(yf), np.zeros_like(zf))
        grad_ze = (np.zeros_like(xf), np.zeros_like(yf), np.ones_like(zf))
        grad_xc = np.round(grad_x, 7)
        grad_yc = np.round(grad_y, 7)
        grad_zc = np.round(grad_z, 7)

        self.assertTrue(np.array_equal(grad_xc, grad_xe))
        self.assertTrue(np.array_equal(grad_yc, grad_ye))
        self.assertTrue(np.array_equal(grad_zc, grad_ze))

    def test_cartesian_grid_gradient_unequal_length_arrays_quadratic(self):
        xm, ym, zm = np.meshgrid(x_short, y, z, indexing='ij')
        xf = xm ** 2
        yf = ym ** 2
        zf = zm ** 2
        grid = gr.CartesianGrid((x_short, y, z))
        grad_x = grid.gradient(xf)
        grad_y = grid.gradient(yf)
        grad_z = grid.gradient(zf)

        grad_xe = (2 * xm, np.zeros_like(yf), np.zeros_like(zf))
        grad_ye = (np.zeros_like(xf), 2 * ym, np.zeros_like(zf))
        grad_ze = (np.zeros_like(xf), np.zeros_like(yf), 2 * zm)

        # Test Float Equality
        grad_xe = np.round(grad_xe, 7)
        grad_ye = np.round(grad_ye, 7)
        grad_ze = np.round(grad_ze, 7)
        grad_xc = np.round(grad_x, 7)
        grad_yc = np.round(grad_y, 7)
        grad_zc = np.round(grad_z, 7)
        self.assertTrue(np.array_equal(grad_xc, grad_xe))
        self.assertTrue(np.array_equal(grad_yc, grad_ye))
        self.assertTrue(np.array_equal(grad_zc, grad_ze))

    def test_cartesian_grid_gradient_container_type(self):
        xm, ym, zm = np.meshgrid(x, y, z, indexing='ij')
        tuple_grid = gr.CartesianGrid((x, y, z))
        list_grid = gr.CartesianGrid([x, y, z])
        tuple_grad = tuple_grid.gradient(xm)
        list_grad = list_grid.gradient(ym)
        self.assertIs(type(tuple_grad), tuple)
        self.assertIs(type(list_grad), list)

    def test_cartesian_grid_divergence_equal_length_arrays_radial_vectors(self):
        xm, ym, zm = np.meshgrid(x, y, z, indexing='ij')
        vector_field = (xm, ym, zm)
        divergence_expected = 3. * np.ones_like(xm)

        grid = gr.CartesianGrid((x, y, z))
        divergence_calculated = grid.divergence(vector_field)

        # Test Float Equality
        divergence_calculated = np.round(divergence_calculated, 7)
        self.assertTrue(np.array_equal(divergence_calculated, divergence_expected))

    def test_cartesian_grid_divergence_equal_length_arrays_azimuthal_vectors(self):
        xm, ym, zm = np.meshgrid(x, y, z, indexing='ij')
        vector_field = (-ym, xm, np.zeros_like(zm))
        divergence_expected = 3. * np.zeros_like(xm)

        grid = gr.CartesianGrid((x, y, z))
        divergence_calculated = grid.divergence(vector_field)

        # Test Float Equality
        divergence_calculated = np.round(divergence_calculated, 7)
        self.assertTrue(np.array_equal(divergence_calculated, divergence_expected))

    def test_cartesian_grid_divergence_unequal_length_arrays_radial_vectors(self):
        xm, ym, zm = np.meshgrid(x_short, y, z, indexing='ij')
        vector_field = (xm, ym, zm)
        divergence_expected = 3. * np.ones_like(xm)

        grid = gr.CartesianGrid((x_short, y, z))
        divergence_calculated = grid.divergence(vector_field)

        # Test Float Equality
        divergence_calculated = np.round(divergence_calculated, 7)
        self.assertTrue(np.array_equal(divergence_calculated, divergence_expected))

    def test_cartesian_grid_divergence_equal_length_arrays_azimuthal_vectors(self):
        xm, ym, zm = np.meshgrid(x_short, y, z, indexing='ij')
        vector_field = (-ym, xm, np.zeros_like(zm))
        divergence_expected = 3. * np.zeros_like(xm)

        grid = gr.CartesianGrid((x_short, y, z))
        divergence_calculated = grid.divergence(vector_field)

        # Test Float Equality
        divergence_calculated = np.round(divergence_calculated, 7)
        self.assertTrue(np.array_equal(divergence_calculated, divergence_expected))

    def test_cartesian_grid_shape(self):
        grid = gr.CartesianGrid((x_short, y, z))
        expected_shape = (len(x_short), len(y), len(z))
        self.assertEqual(grid.shape, expected_shape)


# Test Grid Classes
class TestSphericalGrid(TestCase):
    def test_spherical_grid_init_equal_length_inputs_mesh_equality(self):
        # Initialize Grids with equal length coordinate arrays
        grid = gr.SphericalGrid((r, t, p))
        log_grid = gr.CartesianGrid((r_log, t_log, p_log))

        # Define Expected Behavior
        expected_mesh = np.meshgrid(r, t, p, indexing='ij')
        expected_log_mesh = np.meshgrid(r_log, t_log, p_log, indexing='ij')

        mesh_equality = np.array_equal(grid.mesh, expected_mesh)
        log_mesh_equality = np.array_equal(log_grid.mesh, expected_log_mesh)
        self.assertTrue(mesh_equality)
        self.assertTrue(log_mesh_equality)

    def test_spherical_grid_init_equal_length_inputs_cartesian_equality(self):
        # Initialize Grids with equal length coordinate arrays
        grid = gr.SphericalGrid((r, t, p))
        log_grid = gr.SphericalGrid((r_log, t_log, p_log))

        # Define Expected Behavior
        expected_cart = sp.spherical_to_cartesian(grid.mesh)
        expected_cart = np.round(expected_cart, 7)
        expected_log_cart = sp.spherical_to_cartesian(log_grid.mesh)
        expected_log_cart = np.round(expected_log_cart, 7)

        cart_equality = np.array_equal(np.round(grid.cartesian, 7), expected_cart)
        log_cart_equality = np.array_equal(np.round(log_grid.cartesian, 7), expected_log_cart)
        self.assertTrue(cart_equality)
        self.assertTrue(log_cart_equality)

    def test_spherical_grid_init_unequal_length_inputs_mesh_equality(self):
        # Initialize Grids with equal length coordinate arrays
        mixed_grid = gr.SphericalGrid((r, t_short, p_short))

        # Define Expected Behavior
        expected_mixed_mesh = np.meshgrid(r, t_short, p_short, indexing='ij')

        # Check Equality
        mixed_mesh_eq = np.array_equal(mixed_grid.mesh, expected_mixed_mesh)
        self.assertTrue(mixed_mesh_eq)

    def test_spherical_grid_init_unequal_length_inputs_cartesian_equality(self):
        # Initialize Grids with equal length coordinate arrays
        mixed_grid = gr.SphericalGrid((r, t_short, p_short))

        # Define Expected Behavior
        expected_mixed_cart = sp.spherical_to_cartesian(mixed_grid.mesh)
        expected_mixed_cart = np.round(expected_mixed_cart, 7)

        # Check Equality
        mixed_cart_eq = np.array_equal(np.round(mixed_grid.cartesian, 7), expected_mixed_cart)
        self.assertTrue(mixed_cart_eq)

    def test_spherical_grid_init_list_implicit_container_type(self):
        grid = gr.SphericalGrid([r_short, t_short, p_short])
        container_type = grid._container_type
        coordinate_type = type(grid.coordinates)
        mesh_type = type(grid.mesh)
        cartesian_type = type(grid.cartesian)
        self.assertIs(container_type, list)
        self.assertIs(coordinate_type, list)
        self.assertIs(mesh_type, list)
        self.assertIs(cartesian_type, list)

    def test_spherical_grid_eq_equal_length_grids_return_true(self):
        r2 = 1. * r       # Use this so that x2 == x but is not x
        assert r2 is not r
        assert np.array_equal(r, r2)
        grid = gr.SphericalGrid((r, t, p))
        grid2 = gr.SphericalGrid((r2, t, p))
        self.assertEqual(grid, grid2)

    def test_spherical_grid_eq_very_different_equal_length_grids_return_false(self):
        r2 = 2. * r
        assert not np.array_equal(r, r2)
        grid = gr.SphericalGrid((r, t, p))
        grid2 = gr.CartesianGrid((r2, t, p))
        self.assertNotEqual(grid, grid2)

    def test_spherical_grid_eq_nearly_equal_length_grids_return_false(self):
        r2 = r +1E-7 * np.ones_like(r)
        assert not np.array_equal(r, r2)
        grid = gr.SphericalGrid((r, t, p))
        grid2 = gr.SphericalGrid((r2, t, p))
        self.assertNotEqual(grid, grid2)

    def test_spherical_grid_getitem_equal_length_arrays(self):
        grid = gr.SphericalGrid((r, t, p))
        expected_000 = sp.spherical_to_cartesian((r[0], t[0], p[0]))
        expected_210 = sp.spherical_to_cartesian((r[2], t[1], p[0]))
        self.assertEqual(grid[0, 0, 0], expected_000)
        self.assertEqual(grid[2, 1, 0], expected_210)

    def test_spherical_grid_getitem_unequal_length_arrays(self):
        grid = gr.SphericalGrid((r_short, t, p))
        expected_000 = sp.spherical_to_cartesian((r_short[0], t[0], p[0]))
        expected_210 = sp.spherical_to_cartesian((r_short[2], t[1], p[0]))
        self.assertEqual(grid[0, 0, 0], expected_000)
        self.assertEqual(grid[2, 1, 0], expected_210)

    def test_spherical_grid_getitem_type_list(self):
        grid = gr.SphericalGrid([x, y, z])
        getitem_type = type(grid[0, 0, 0])
        self.assertIs(getitem_type, list)
        self.assertIs(getitem_type, grid._container_type)

    def test_spherical_grid_mesh_item_equal_length_arrays(self):
        grid = gr.SphericalGrid((r, t, p))
        expected_000 = (r[0], t[0], p[0])
        expected_210 = (r[2], t[1], p[0])
        self.assertEqual(grid.mesh_item(0, 0, 0), expected_000)
        self.assertEqual(grid.mesh_item(2, 1, 0), expected_210)

    def test_spherical_grid_mesh_item_unequal_length_arrays(self):
        grid = gr.SphericalGrid((r_short, t, p))
        expected_000 = (r_short[0], t[0], p[0])
        expected_210 = (r_short[2], t[1], p[0])
        self.assertEqual(grid.mesh_item(0, 0, 0), expected_000)
        self.assertEqual(grid.mesh_item(2, 1, 0), expected_210)

    def test_spherical_grid_mesh_item_type_list(self):
        grid = gr.SphericalGrid([r, t, p])
        getitem_type = type(grid.mesh_item(0, 0, 0))
        self.assertIs(getitem_type, list)
        self.assertIs(getitem_type, grid._container_type)

    def test_spherical_grid_iter_tuples(self):
        grid = gr.SphericalGrid((r, t, p))
        xm, ym, zm = sp.spherical_to_cartesian(np.meshgrid(r, t, p, indexing='ij'))
        x_expected = xm.flatten()
        y_expected = ym.flatten()
        z_expected = zm.flatten()
        for c, xe, ye, ze in zip(grid, x_expected, y_expected, z_expected):
            self.assertEqual(c, (xe, ye, ze))

    def test_spherical_grid_iter_lists(self):
        grid = gr.SphericalGrid([r, t, p])
        xm, ym, zm = sp.spherical_to_cartesian(np.meshgrid(r, t, p, indexing='ij'))
        x_expected = xm.flatten()
        y_expected = ym.flatten()
        z_expected = zm.flatten()
        for c, xe, ye, ze in zip(grid, x_expected, y_expected, z_expected):
            self.assertEqual(c, [xe, ye, ze])

    def test_spherical_grid_iter_types(self):
        tuple_grid = gr.SphericalGrid((r, t, p))
        list_grid = gr.SphericalGrid([r, t, p])
        for tup, l in zip(tuple_grid, list_grid):
            self.assertIs(type(tup), tuple)
            self.assertIs(type(l), list)

    def test_spherical_grid_mesh_iter_tuples(self):
        grid = gr.SphericalGrid((r, t, p))
        rm, tm, pm = np.meshgrid(r, t, p, indexing='ij')
        r_expected = rm.flatten()
        t_expected = tm.flatten()
        p_expected = pm.flatten()
        for c, re, te, pe in zip(grid.mesh_iter(), r_expected, t_expected, p_expected):
            self.assertEqual(c, (re, te, pe))

    def test_spherical_grid_mesh_iter_lists(self):
        grid = gr.SphericalGrid([r, t, p])
        rm, tm, pm = np.meshgrid(r, t, p, indexing='ij')
        r_expected = rm.flatten()
        t_expected = tm.flatten()
        p_expected = pm.flatten()
        for c, re, te, pe in zip(grid.mesh_iter(), r_expected, t_expected, p_expected):
            self.assertEqual(c, [re, te, pe])

    def test_spherical_grid_mesh_iter_types(self):
        tuple_grid = gr.SphericalGrid((r, t, p))
        list_grid = gr.SphericalGrid([r, t, p])
        for tup, l in zip(tuple_grid.mesh_iter(), list_grid.mesh_iter()):
            self.assertIs(type(tup), tuple)
            self.assertIs(type(l), list)

    def test_spherical_grid_volume_equal_length_arrays(self):
        grid = gr.SphericalGrid((r, t, p))
        dr, dt, dp = 0.05 * np.ones_like(r), 0.05 * pi *np.ones_like(t), 0.1 * pi * np.ones_like(p)
        dr[0] = 0.5 * dr[0]
        dr[-1] = 0.5 * dr[-1]
        dt[0] = 0.5 * dt[0]
        dt[-1] = 0.5 * dt[-1]
        dp[0] = 0.5 * dp[0]
        dp[-1] = 0.5 * dp[-1]
        drg, dtg, dpg = np.meshgrid(dr, dt, dp, indexing='ij')
        rg, tg, pg = np.meshgrid(r, t, p, indexing='ij')
        v_e = sp.calculate_spherical_volume_element(rg, tg, (drg, dtg, dpg))

        # Round For float equality check
        v_e = np.round(v_e, 7)
        v_c = np.round(grid.volume, 7)
        self.assertTrue(np.array_equal(v_e, v_c))

    def test_spherical_grid_volume_unequal_length_arrays(self):
        grid = gr.SphericalGrid((r_short, t, p))
        dr, dt, dp = 0.1 * np.ones_like(r_short), 0.05 * pi * np.ones_like(t), 0.1 * pi * np.ones_like(p)
        dr[0] = 0.5 * dr[0]
        dr[-1] = 0.5 * dr[-1]
        dt[0] = 0.5 * dt[0]
        dt[-1] = 0.5 * dt[-1]
        dp[0] = 0.5 * dp[0]
        dp[-1] = 0.5 * dp[-1]
        drg, dtg, dpg = np.meshgrid(dr, dt, dp, indexing='ij')
        rg, tg, pg = np.meshgrid(r_short, t, p, indexing='ij')
        v_e = sp.calculate_spherical_volume_element(rg, tg, (drg, dtg, dpg))

        # Round For float equality check
        v_e = np.round(v_e, 7)
        v_c = np.round(grid.volume, 7)
        self.assertTrue(np.array_equal(v_e, v_c))

    def test_spherical_grid_gradient_equal_length_arrays_linear(self):
        rm, tm, pm = np.meshgrid(r, t, p, indexing='ij')
        rf = rm
        grid = gr.SphericalGrid((r, t, p))
        grad_r = grid.gradient(rf)

        grad_re = (np.ones_like(rf), np.zeros_like(tm), np.zeros_like(pm))
        grad_rc = np.round(grad_r, 6)

        self.assertTrue(np.array_equal(grad_rc, grad_re))

    def test_spherical_grid_gradient_unequal_length_arrays_linear(self):
        rm, tm, pm = np.meshgrid(r_short, t, p, indexing='ij')
        rf = rm
        grid = gr.SphericalGrid((r_short, t, p))
        grad_r = grid.gradient(rf)

        grad_re = (np.ones_like(rf), np.zeros_like(tm), np.zeros_like(pm))
        grad_rc = np.round(grad_r, 6)

        self.assertTrue(np.array_equal(grad_rc, grad_re))

    def test_spherical_grid_gradient_unequal_length_arrays_quadratic(self):
        rm, tm, pm = np.meshgrid(r, t, p, indexing='ij')
        rf = rm ** 2
        grid = gr.SphericalGrid((r, t, p))
        grad_r = grid.gradient(rf)

        grad_re = np.round((2 * rm, np.zeros_like(tm), np.zeros_like(pm)), 6)
        grad_rc = np.round(grad_r, 6)

        test = grad_re == grad_rc
        t0 = np.sum(test[0])
        t1 = np.sum(test[1])
        t2 = np.sum(test[2])

        self.assertTrue(np.array_equal(grad_rc, grad_re))

    def test_spherical_grid_gradient_container_type(self):
        xm, ym, zm = np.meshgrid(r, t, p, indexing='ij')
        tuple_grid = gr.SphericalGrid((r, t, p))
        list_grid = gr.SphericalGrid([r, t, p])
        tuple_grad = tuple_grid.gradient(xm)
        list_grad = list_grid.gradient(ym)
        self.assertIs(type(tuple_grad), tuple)
        self.assertIs(type(list_grad), list)

    def test_spherical_grid_divergence_equal_length_arrays_radial_vectors(self):
        rl = r + 1E-12 * np.ones_like(r)
        tl = t + 1E-12 * np.ones_like(t)
        pl = r + 1E-12 * np.ones_like(p)
        rm, tm, pm = np.meshgrid(rl, tl, pl, indexing='ij')
        vector_field = (rm, np.zeros_like(tm), np.zeros_like(pm))
        divergence_expected = 3. * np.ones_like(rm)

        grid = gr.SphericalGrid((rl, tl, pl))
        divergence_calculated = grid.divergence(vector_field)

        # Test Float Equality
        divergence_calculated = np.round(divergence_calculated, 7)
        self.assertTrue(np.array_equal(divergence_calculated, divergence_expected))

    def test_cartesian_grid_divergence_equal_length_arrays_azimuthal_vectors(self):
        rl = r + 1E-12 * np.ones_like(r)
        tl = t + 1E-12 * np.ones_like(t)
        pl = r + 1E-12 * np.ones_like(p)
        rm, tm, pm = np.meshgrid(rl, tl, pl, indexing='ij')
        vector_field = (np.zeros_like(rm), np.zeros_like(tm), rm * np.sin(tm))
        divergence_expected = np.zeros_like(rm)

        grid = gr.SphericalGrid((rl, tl, pl))
        divergence_calculated = grid.divergence(vector_field)

        # Test Float Equality
        divergence_calculated = np.round(divergence_calculated, 7)
        self.assertTrue(np.array_equal(divergence_calculated, divergence_expected))

    def test_spherical_grid_divergence_unequal_length_arrays_radial_vectors(self):
        rl = r_short + 1E-12 * np.ones_like(r_short)
        tl = t + 1E-12 * np.ones_like(t)
        pl = r + 1E-12 * np.ones_like(p)
        rm, tm, pm = np.meshgrid(rl, tl, pl, indexing='ij')
        vector_field = (rm, np.zeros_like(tm), np.zeros_like(pm))
        divergence_expected = 3. * np.ones_like(rm)

        grid = gr.SphericalGrid((rl, tl, pl))
        divergence_calculated = grid.divergence(vector_field)

        # Test Float Equality
        divergence_calculated = np.round(divergence_calculated, 7)
        self.assertTrue(np.array_equal(divergence_calculated, divergence_expected))

    def test_cartesian_grid_divergence_unequal_length_arrays_azimuthal_vectors(self):
        rl = r_short + 1E-12 * np.ones_like(r_short)
        tl = t + 1E-12 * np.ones_like(t)
        pl = r + 1E-12 * np.ones_like(p)
        rm, tm, pm = np.meshgrid(rl, tl, pl, indexing='ij')
        vector_field = (np.zeros_like(rm), np.zeros_like(tm), rm * np.sin(tm))
        divergence_expected = np.zeros_like(rm)

        grid = gr.SphericalGrid((rl, tl, pl))
        divergence_calculated = grid.divergence(vector_field)

        # Test Float Equality
        divergence_calculated = np.round(divergence_calculated, 7)
        self.assertTrue(np.array_equal(divergence_calculated, divergence_expected))

    def test_spherical_grid_shape(self):
        grid = gr.CartesianGrid((r_short, t, p))
        expected_shape = (len(r_short), len(t), len(p))
        self.assertEqual(grid.shape, expected_shape)


class TestDataGrid(TestCase):
    def test_data_grid_init_cartesian_grid(self):
        data = np.ones_like(c_grid.mesh[0])
        data_grid = gr.DataGrid(data, c_grid)
        self.assertTrue(np.array_equal(data_grid.data, data))
        self.assertEqual(data_grid.grid, c_grid)
        self.assertIs(data_grid.grid, c_grid)

    def test_data_grid_init_spherical_grid(self):
        data = np.ones_like(s_grid.mesh[0])
        data_grid = gr.DataGrid(data, s_grid)
        self.assertTrue(np.array_equal(data_grid.data, data))
        self.assertEqual(data_grid.grid, s_grid)
        self.assertIs(data_grid.grid, s_grid)

    def test_data_grid_init_raises_unequal_shape_assertion_error(self):
        data = np.ones_like(short_s_grid.mesh[0])
        with self.assertRaises(AssertionError):
            gr.DataGrid(data, s_grid)

    def test_data_grid_eq_returns_equality(self):
        data_1 = np.ones_like(s_grid.mesh[0])
        data_2 = np.ones(s_grid.shape)
        data_grid_1 = gr.DataGrid(data_1, s_grid)
        data_grid_2 = gr.DataGrid(data_2, s_grid)
        self.assertEqual(data_grid_1, data_grid_2)

    def test_data_grid_eq_returns_not_equality_different_data(self):
        data_1 = np.ones_like(s_grid.mesh[0])
        data_2 = np.zeros(s_grid.shape)
        data_grid_1 = gr.DataGrid(data_1, s_grid)
        data_grid_2 = gr.DataGrid(data_2, s_grid)
        self.assertNotEqual(data_grid_1, data_grid_2)

    def test_data_grid_eq_returns_not_equality_different_grids(self):
        data_1 = np.ones_like(s_grid.mesh[0])
        data_2 = np.ones(s_grid.shape)
        data_grid_1 = gr.DataGrid(data_1, s_grid)
        data_grid_2 = gr.DataGrid(data_2, c_grid)
        self.assertNotEqual(data_grid_1, data_grid_2)

    def test_grid_getitem(self):
        data = np.ones(s_grid.shape)
        data_grid = gr.DataGrid(data, s_grid)
        expected_000 = (1, s_grid[0, 0, 0])
        expected_210 = (1, s_grid[2, 1, 0])
        self.assertEqual(data_grid[0, 0, 0], expected_000)
        self.assertEqual(data_grid[2, 1, 0], expected_210)

    def test_grid_meshitem(self):
        data = np.ones(s_grid.shape)
        data_grid = gr.DataGrid(data, s_grid)
        expected_000 = (1, s_grid.mesh_item(0, 0, 0))
        expected_210 = (1, s_grid.mesh_item(2, 1, 0))
        self.assertEqual(data_grid.mesh_item(0, 0, 0), expected_000)
        self.assertEqual(data_grid.mesh_item(2, 1, 0), expected_210)

    def test_data_grid_iter(self):
        data = np.ones(s_grid.shape)
        data_grid = gr.DataGrid(data, s_grid)
        for dg, g in zip(data_grid, s_grid):
            self.assertEqual(dg, (1, g))

    def test_data_grid_mesh_iter(self):
        data = np.ones(s_grid.shape)
        data_grid = gr.DataGrid(data, s_grid)
        for dg, g in zip(data_grid.mesh_iter(), s_grid.mesh_iter()):
            self.assertEqual(dg, (1, g))

    def test_data_grid_add(self):
        data_1 = np.ones(s_grid.shape)
        data_2 = 2 * data_1
        data_3 = data_1 + data_2
        data_grid_1 = gr.DataGrid(data_1, s_grid)
        data_grid_2 = gr.DataGrid(data_2, s_grid)
        data_grid_3_e = gr.DataGrid(data_3, s_grid)
        data_grid_3_c = data_grid_1 + data_grid_2
        self.assertEqual(data_grid_3_c, data_grid_3_e)

    def test_data_grid_add_raises_unequal_grid_exception(self):
        data_1 = np.ones(s_grid.shape)
        data_2 = 2 * data_1
        data_grid_1 = gr.DataGrid(data_1, s_grid)
        data_grid_2 = gr.DataGrid(data_2, c_grid)
        with self.assertRaises(AssertionError):
            data_grid_3 = data_grid_1 + data_grid_2

    def test_data_grid_mul(self):
        data_1 = np.ones(s_grid.shape)
        data_2 = 2 * data_1
        data_3 = data_1 * data_2
        data_grid_1 = gr.DataGrid(data_1, s_grid)
        data_grid_2 = gr.DataGrid(data_2, s_grid)
        data_grid_3_e = gr.DataGrid(data_3, s_grid)
        data_grid_3_c = data_grid_1 * data_grid_2
        self.assertEqual(data_grid_3_c, data_grid_3_e)

    def test_data_grid_sub(self):
        data_1 = np.ones(s_grid.shape)
        data_2 = 2 * data_1
        data_3 = data_1 - data_2
        data_grid_1 = gr.DataGrid(data_1, s_grid)
        data_grid_2 = gr.DataGrid(data_2, s_grid)
        data_grid_3_e = gr.DataGrid(data_3, s_grid)
        data_grid_3_c = data_grid_1 - data_grid_2
        self.assertEqual(data_grid_3_c, data_grid_3_e)

    def test_data_grid_abs_cartesian(self):
        data = np.ones(c_grid.shape)
        data_grid = gr.DataGrid(data, c_grid)
        self.assertEqual(8, abs(data_grid))

    def test_data_grid_abs_spherical(self):
        data = np.ones(s_grid.shape)
        data_grid = gr.DataGrid(data, s_grid)
        vol = 4 * pi/3
        self.assertAlmostEqual(vol, abs(data_grid), 2) # Note 1% numerical error


class TestVectorGrid(TestCase):
    def test_vector_grid_init_from_cartesian_operator_mesh(self):
        ops = []
        for xx in x:
            for yy in y:
                for zz in z:
                    ops.append(la.CartesianSpinOperator((0, xx, yy, zz)))
        ops = np.reshape(ops, c_grid.shape)
        vector_grid = gr.VectorGrid(ops, c_grid)
        self.assertIs(vector_grid._operator_type, la.CartesianSpinOperator)
        self.assertIs(vector_grid._container_type, tuple)
        self.assertEqual(vector_grid._operator_dim, 4)
        self.assertTrue(np.array_equal(ops, vector_grid.data))
        self.assertEqual(vector_grid[0, 0, 0], (la.CartesianSpinOperator((0, -1, -1, -1)), (-1, -1, -1)))

    def test_vector_grid_init_from_spherical_operator_mesh(self):
        ops = []
        for xx in r:
            for yy in t:
                for zz in p:
                    ops.append(la.SphericalSpinOperator((0, xx, yy, zz)))
        ops = np.reshape(ops, s_grid.shape)
        vector_grid = gr.VectorGrid(ops, s_grid)
        self.assertIs(vector_grid._operator_type, la.SphericalSpinOperator)
        self.assertIs(vector_grid._container_type, tuple)
        self.assertEqual(vector_grid._operator_dim, 4)
        self.assertTrue(np.array_equal(ops, vector_grid.data))
        self.assertEqual(vector_grid[0, 0, 0], (la.SphericalSpinOperator((0, 0, 0, 0)), (0, 0, 0)))

    def test_vector_grid_init_from_cartesian_coordinate_mesh(self):
        ops_e = []
        for xx in x:
            for yy in y:
                for zz in z:
                    ops_e.append(la.CartesianSpinOperator((0, xx, yy, zz)))
        ops_e = np.reshape(ops_e, c_grid.shape)
        vector_grid = gr.VectorGrid(c_grid.mesh, c_grid, operator_type=la.CartesianSpinOperator)
        self.assertIs(vector_grid._operator_type, la.CartesianSpinOperator)
        self.assertIs(vector_grid._container_type, tuple)
        self.assertEqual(vector_grid._operator_dim, 4)
        self.assertTrue(np.array_equal(ops_e, vector_grid.data))
        self.assertEqual(vector_grid[0, 0, 0], (la.CartesianSpinOperator((0, -1, -1, -1)), (-1, -1, -1)))

    def test_vector_grid_init_from_spherical_coordinate_mesh(self):
        ops_e = []
        for xx in r:
            for yy in t:
                for zz in p:
                    ops_e.append(la.SphericalSpinOperator((0, xx, yy, zz)))
        ops_e = np.reshape(ops_e, s_grid.shape)
        vector_grid = gr.VectorGrid(list(s_grid.mesh), s_grid, operator_type=la.SphericalSpinOperator)
        self.assertIs(vector_grid._operator_type, la.SphericalSpinOperator)
        self.assertIs(vector_grid._container_type, list)
        self.assertEqual(vector_grid._operator_dim, 4)
        self.assertTrue(np.array_equal(ops_e, vector_grid.data))
        self.assertEqual(vector_grid[0, 0, 0], (la.SphericalSpinOperator((0, 0, 0, 0)), (0, 0, 0)))

    def test_vector_grid_vectors(self):
        ops = []
        for xx in r:
            for yy in t:
                for zz in p:
                    ops.append(la.SphericalSpinOperator((0, xx, yy, zz)))
        ops = np.reshape(ops, s_grid.shape)
        vector_grid = gr.VectorGrid(ops, s_grid)
        vectors = vector_grid.vectors
        self.assertTrue(np.array_equal(vectors, s_grid.mesh))

    def test_vector_grid_cartesian_vectors(self):
        ops = []
        for xx in r:
            for yy in t:
                for zz in p:
                    ops.append(la.SphericalSpinOperator((0, xx, yy, zz)))
        ops = np.reshape(ops, s_grid.shape)
        vector_grid = gr.VectorGrid(ops, s_grid)
        vectors = vector_grid.cartesian_vectors
        self.assertTrue(np.array_equal(vectors, s_grid.cartesian))


class TestDataGridCalculus(TestCase):
    def test_spherical_gradient(self):
        funct = s_grid.mesh[0]
        data_grid = gr.DataGrid(funct, s_grid)
        ops = []
        for xx in r:
            for yy in t:
                for zz in p:
                    ops.append(la.SphericalSpinOperator((0, 1, 0, 0)))
        ops = np.reshape(ops, s_grid.shape)
        vector_grid = gr.VectorGrid(ops, s_grid)
        self.assertAlmostEqual(vector_grid, data_grid.gradient(la.SphericalSpinOperator, tuple))

    def test_spherical_radial_divergence(self):
        v_funct = (s_grid.mesh[0], np.zeros(s_grid.shape), np.zeros(s_grid.shape))
        vector_grid = gr.VectorGrid(v_funct, s_grid, operator_type=la.SphericalSpinOperator)
        expected_data = 3 * np.ones(s_grid.shape)
        data_grid = gr.DataGrid(expected_data, s_grid)
        self.assertAlmostEqual(data_grid, vector_grid.divergence())



if __name__ == '__main__':
    unittest.main()
