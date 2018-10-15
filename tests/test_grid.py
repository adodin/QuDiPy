import unittest
from unittest import TestCase
import QuDiPy.containers.grids as gr
import QuDiPy.coordinates.cartesian
import QuDiPy.coordinates.spherical as sp
import numpy as np
from numpy import pi
import QuDiPy.math.linear_algebra as la


# Test Cartesian Calculus Functions (Spherical in Test Spherical)
class TestCalculateDifferentials(TestCase):
    def test_calculate_uniformly_spaced_differentials(self):
        x = np.linspace(-1., 1., 11)
        y = np.linspace(-1., 1., 21)
        z = np.linspace(0., 1., 11)
        dx = 0.2 * np.ones_like(x)
        dx[0] = 0.1
        dx[-1] = 0.1
        dy = 0.1 * np.ones_like(y)
        dy[0] = 0.05
        dy[-1] = 0.05
        dz = 0.1 * np.ones_like(z)
        dz[0] = 0.05
        dz[-1] = 0.05
        correct_differentials = (dx, dy, dz)
        calculated_diffs = QuDiPy.coordinates.cartesian.calculate_differentials((x, y, z))
        for correct, calc in zip(correct_differentials, calculated_diffs):
            rounded_calc = np.round(calc, 7)
            self.assertTrue(np.array_equal(rounded_calc, correct))

    def test_calculate_nonuniformly_spaced_differentials(self):
        x = (0., 0.1, 0.3, 0.9)
        dx = (0.05, 0.15, 0.4, 0.3)
        correct_differentials = (dx,)
        calculated_diffs = QuDiPy.coordinates.cartesian.calculate_differentials((x,))
        for correct, calc in zip(correct_differentials, calculated_diffs):
            rounded_calc = np.round(calc, 7)
            self.assertTrue(np.array_equal(rounded_calc, correct))


class TestCalculateCartesianVolumeElement(TestCase):
    def test_calculate_cartesian_volume_element_uniform(self):
        x = np.linspace(-1., 1., 11)
        y = np.linspace(-1., 1., 21)
        z = np.linspace(0., 1., 11)
        dx = 0.2 * np.ones_like(x)
        dx[0] = 0.1
        dx[-1] = 0.1
        dy = 0.1 * np.ones_like(y)
        dy[0] = 0.05
        dy[-1] = 0.05
        dz = 0.1 * np.ones_like(z)
        dz[0] = 0.05
        dz[-1] = 0.05
        dx_grid, dy_grid, dz_grid = np.meshgrid(dx, dy, dz, indexing='ij')
        correct_volume = dx_grid * dy_grid * dz_grid
        calculated_volume = QuDiPy.coordinates.cartesian.calculate_cartesian_volume_element((x, y, z))
        self.assertTrue(np.array_equal(np.round(calculated_volume, 7), np.round(correct_volume, 7)))


class TestCalculateCartesianGradient(TestCase):
    def test_calculate_cartesian_gradient_linear(self):
        x = np.linspace(-1., 1., 11)
        y = np.linspace(-1., 1., 21)
        z = np.linspace(0., 1., 11)
        x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing='ij')
        funct = z_grid
        correct_gradient = (np.zeros_like(x_grid), np.zeros_like(y_grid), np.ones_like(z_grid))
        calculated_gradient = QuDiPy.coordinates.cartesian.calculate_cartesian_gradient(funct, (x, y, z))
        for correct, calculated in zip(correct_gradient, calculated_gradient):
            self.assertTrue(np.array_equal(np.round(calculated, 7), correct))

    def test_calculate_cartesian_gradient_xy(self):
        x = np.linspace(-1., 1., 11)
        y = np.linspace(-1., 1., 21)
        z = np.linspace(0., 1., 11)
        x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing='ij')
        funct = x_grid * y_grid
        correct_gradient = (y_grid, x_grid, np.zeros_like(z_grid))
        calculated_gradient = QuDiPy.coordinates.cartesian.calculate_cartesian_gradient(funct, (x, y, z))
        for correct, calculated in zip(correct_gradient, calculated_gradient):
            self.assertTrue(np.array_equal(np.round(calculated, 7), np.round(correct, 7)))

    def test_calculate_cartesian_gradient_quadratic(self):
        x = np.linspace(-1., 1., 11)
        y = np.linspace(-1., 1., 21)
        z = np.linspace(0., 1., 11)
        x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing='ij')
        funct = z_grid ** 2
        correct_gradient = (np.zeros_like(x_grid), np.zeros_like(y_grid), 2 * z_grid)
        calculated_gradient = QuDiPy.coordinates.cartesian.calculate_cartesian_gradient(funct, (x, y, z))
        for correct, calculated in zip(correct_gradient, calculated_gradient):
            self.assertTrue(np.array_equal(np.round(calculated, 7), np.round(correct, 7)))


class TestCalculateCartesianDivergence(TestCase):
    def test_calculate_cartesian_divergence_whirlpool(self):
        x = np.linspace(-1., 1., 11)
        y = np.linspace(-1., 1., 21)
        z = np.linspace(0., 1., 11)
        x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing='ij')
        funct = (y_grid, x_grid, np.zeros_like(z_grid))
        correct_divergence = np.zeros_like(x_grid)
        calculated_divergence = QuDiPy.coordinates.cartesian.calculate_cartesian_divergence(funct, (x, y, z))
        self.assertTrue(np.array_equal(np.round(calculated_divergence, 7), correct_divergence))

    def test_calculate_cartesian_divergence_radial(self):
        x = np.linspace(-1., 1., 11)
        y = np.linspace(-1., 1., 21)
        z = np.linspace(0., 1., 11)
        x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing='ij')
        funct = QuDiPy.coordinates.cartesian.calculate_cartesian_gradient(.5 * (x_grid ** 2 + y_grid ** 2 + z_grid ** 2), (x, y, z))
        correct_divergence = 3*np.ones_like(x_grid)
        calculated_divergence = QuDiPy.coordinates.cartesian.calculate_cartesian_divergence(funct, (x, y, z))
        self.assertTrue(np.array_equal(np.round(calculated_divergence, 7), correct_divergence))


class TestVectorizeOperatorGrid(TestCase):
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
        ops = np.array([[[s000, s001], [s010, s011]],
                        [[s100, s101], [s110, s111]]])
        o_grid_ex = gr.GridData(ops, grid)
        o_grid_ac = gr.initialize_operator_grid(grid, la.CartesianSpinOperator, i_coord=0.)
        self.assertEqual(o_grid_ac, o_grid_ex)


# Test Grid Objects
class TestGrid(TestCase):
    def test_grid_equality_override(self):
        x = (0., 1., 2.)
        x2 = (0., 1., 3.)
        x3 = (0., 2.)
        xprime = (0., 1., 4./2)
        y = (0., 5.)
        z = (1., 2., 3., 4.)
        grid1 = gr.CartesianGrid((x, y, z))
        grid1priime = gr.CartesianGrid((xprime, y, z))
        grid2 = gr.CartesianGrid((x2, y, z))
        grid3 = gr.CartesianGrid((x3, y, z))
        self.assertEqual(grid1, grid1priime)
        self.assertNotEqual(grid1, grid2)
        self.assertNotEqual(grid1, grid3)
        self.assertNotEqual(grid2, grid3)

    def test_grid_getitem_override(self):
        x1 = (1., 2., 3.)
        x2 = (100, 200, 300)
        grid1d = gr.CartesianGrid((x1,))
        grid2d = gr.CartesianGrid((x1, x2))
        self.assertEqual(grid1d[0], (1.,))
        self.assertEqual(grid1d[2], (3.,))
        self.assertEqual(grid2d[0, 0], (1., 100))
        self.assertEqual(grid2d[0, 1], (1., 200))

    def test_grid_iter_method(self):
        x1 = (1., 2., 3.)
        x1repeat = (1., 1., 1., 2., 2., 2., 3., 3., 3.)
        x2 = (100, 200, 300)
        x2repeat = (100, 200, 300, 100, 200, 300, 100, 200, 300)
        grid1d = gr.CartesianGrid((x1,))
        grid2d = gr.CartesianGrid((x1, x2))
        for gt, xt in zip(grid1d, x1):
            self.assertEqual(gt, (xt,))
        for g2, x1t, x2t in zip(grid2d, x1repeat, x2repeat):
            self.assertEqual(g2, (x1t, x2t))

    def test_grid_shape(self):
        x = (1., 2., 3.)
        y = (1., 2., 3., 4.)
        z = (5., 1.)
        grid = gr.CartesianGrid((x, y, z))
        self.assertEqual(grid.shape, (3, 4, 2))


class TestGridData(TestCase):
    def test_grid_data_getitem_overload_1D_data(self):
        x1 = (1., 2.)
        x2 = (100, 200)
        grid2d = gr.CartesianGrid((x1, x2))
        data1d = np.array([['a', 'b'], ['c', 'd']])
        d1d2 = grid2d.grid[0] + grid2d.grid[1]
        g_d = gr.GridData(data1d, grid2d)
        g_d2 = gr.GridData(d1d2, grid2d)
        self.assertEqual(('a', (1., 100)), g_d[0, 0])
        self.assertEqual(('c', (2., 100)), g_d[1, 0])
        self.assertEqual(('b', (1., 200)), g_d[0, 1])
        self.assertEqual(('d', (2., 200)), g_d[1, 1])
        self.assertEqual((101, (1., 100)), g_d2[0, 0])
        self.assertEqual((102, (2., 100)), g_d2[1, 0])
        self.assertEqual((201, (1., 200)), g_d2[0, 1])
        self.assertEqual((202, (2., 200)), g_d2[1, 1])

    def test_grid_data_getitem_overload_2D_data(self):
        x1 = (1., 2.)
        x2 = (100, 200)
        grid2d = gr.CartesianGrid((x1, x2))
        data2d = (np.array([['a', 'b'], ['c', 'd']]), np.array([[1, 2], [3, 4]]))
        g_d = gr.GridData(data2d, grid2d)
        self.assertEqual((('a', 1), (1., 100)), g_d[0, 0])

    def test_grid_data_iter_overload_1D_data(self):
        x1 = (1., 2.)
        x1r = (1., 1., 2., 2.)
        x2 = (100, 200)
        x2r = (100, 200, 100, 200)
        data1d = np.array([['a', 'b'], ['c', 'd']])
        data1d_flattened = data1d.flatten()
        grid2d = gr.CartesianGrid((x1, x2))
        g_d = gr.GridData(data1d, grid2d)
        for gdt, de, ge in zip(g_d, data1d_flattened, grid2d):
            self.assertEqual((de, ge), gdt)
        for gdt, de, ge1, ge2 in zip(g_d, data1d_flattened, x1r, x2r):
            self.assertEqual((de, (ge1, ge2)), gdt)

    def test_grid_data_iter_overload_2D_data(self):
        x1 = (1., 2.)
        x2 = (100, 200)
        data2d = (np.array([['a', 'b'], ['c', 'd']]), np.array([[1, 2], [3, 4]]))
        data1f = data2d[0].flatten()
        data2f = data2d[1].flatten()
        grid2d = gr.CartesianGrid((x1, x2))
        g_d = gr.GridData(data2d, grid2d)
        for gdt, de1, de2, ge in zip(g_d, data1f, data2f, grid2d):
            self.assertEqual(((de1, de2), ge), gdt)

    def test_grid_data_sum_1D_data(self):
        x1 = (1., 2.)
        x2 = (10, 20)
        grid = gr.CartesianGrid((x1, x2))
        d1 = np.array([[1., 2.], [3., 4.]])
        d2 = np.array([[10., 20.], [30., 40.]])
        d_sum = d1 + d2
        expected = gr.GridData(d_sum, grid)
        actual = gr.GridData(d1, grid) + gr.GridData(d2, grid)
        self.assertEqual(expected, actual)

    def test_grid_data_sum_2D_data(self):
        x1 = (1., 2.)
        x2 = (10, 20)
        grid = gr.CartesianGrid((x1, x2))
        d1 = np.array([[1., 2.], [3., 4.]])
        d2 = np.array([[10., 20.], [30., 40.]])
        d_sum_0 = d1 + d2
        d_sum_1 = 2*d2 - d1
        expected = gr.GridData((d_sum_0, d_sum_1), grid)
        actual = gr.GridData((d1, -1*d1), grid) + gr.GridData((d2, 2*d2), grid)
        self.assertEqual(expected, actual)

    def test_grid_data_mul_1D_data(self):
        x1 = (1., 2.)
        x2 = (10, 20)
        grid = gr.CartesianGrid((x1, x2))
        d1 = np.array([[1., 2.], [3., 4.]])
        d2 = np.array([[10., 20.], [30., 40.]])
        d_sum = d1 * d2
        expected = gr.GridData(d_sum, grid)
        actual = gr.GridData(d1, grid) * gr.GridData(d2, grid)
        self.assertEqual(expected, actual)

    def test_grid_data_mul_2D_data(self):
        x1 = (1., 2.)
        x2 = (10, 20)
        grid = gr.CartesianGrid((x1, x2))
        d1 = np.array([[1., 2.], [3., 4.]])
        d2 = np.array([[10., 20.], [30., 40.]])
        d_sum_0 = d1 * d2
        d_sum_1 = 2 * d2 * -1 * d1
        expected = gr.GridData((d_sum_0, d_sum_1), grid)
        actual = gr.GridData((d1, -1 * d1), grid) * gr.GridData((d2, 2 * d2), grid)
        self.assertEqual(expected, actual)

    def test_grid_data_diff_1D_data(self):
        x1 = (1., 2.)
        x2 = (10, 20)
        grid = gr.CartesianGrid((x1, x2))
        d1 = np.array([[1., 2.], [3., 4.]])
        d2 = np.array([[10., 20.], [30., 40.]])
        d_sum = d1 - d2
        expected = gr.GridData(d_sum, grid)
        actual = gr.GridData(d1, grid) - gr.GridData(d2, grid)
        self.assertEqual(expected, actual)

    def test_grid_data_diff_2D_data(self):
        x1 = (1., 2.)
        x2 = (10, 20)
        grid = gr.CartesianGrid((x1, x2))
        d1 = np.array([[1., 2.], [3., 4.]])
        d2 = np.array([[10., 20.], [30., 40.]])
        d_sum_0 = d1 - d2
        d_sum_1 = -2 * d2 - d1
        expected = gr.GridData((d_sum_0, d_sum_1), grid)
        actual = gr.GridData((d1, -1 * d1), grid) - gr.GridData((d2, 2 * d2), grid)
        self.assertEqual(expected, actual)

    def test_grid_data_rmul_1D_data(self):
        x1 = (1., 2.)
        x2 = (10, 20)
        grid = gr.CartesianGrid((x1, x2))
        d1 = np.array([[1., 2.], [3., 4.]])
        d_sum = 2*d1
        expected = gr.GridData(d_sum, grid)
        actual = 2*gr.GridData(d1, grid)
        self.assertEqual(expected, actual)

    def test_grid_data_rmul_2D_data(self):
        x1 = (1., 2.)
        x2 = (10, 20)
        grid = gr.CartesianGrid((x1, x2))
        d1 = np.array([[1., 2.], [3., 4.]])
        d2 = np.array([[10., 20.], [30., 40.]])
        d_sum_0 = 2 * d1
        d_sum_1 = 2 * d2
        expected = gr.GridData((d_sum_0, d_sum_1), grid)
        actual = 2 * gr.GridData((d1, d2), grid)
        self.assertEqual(expected, actual)

    def test_grid_data_like(self):
        x1 = (1., 2.)
        x2 = (100, 200)
        data2d = (np.array([['a', 'b'], ['c', 'd']]), np.array([[1, 2], [3, 4]]))
        grid2d = gr.CartesianGrid((x1, x2))
        g_d = gr.GridData(data2d, grid2d)
        data2d_other = (np.array([['e', 'f'], ['g', 'h']]), np.array([[10, 20], [30, 40]]))
        g_expected = gr.GridData(data2d_other, grid2d)
        g_test = g_d.like(data2d_other)
        self.assertEqual(g_test, g_expected)


class TestCartesianGrid(TestCase):
    def test_cartesian_grid_init(self):
        x = np.linspace(-1., 1., 11)
        y = np.linspace(-1., 1., 21)
        z = np.linspace(0., 1., 11)
        correct_grid = np.meshgrid(x, y, z, indexing='ij')
        dx = 0.2 * np.ones_like(x)
        dx[0] = 0.1
        dx[-1] = 0.1
        dy = 0.1 * np.ones_like(y)
        dy[0] = 0.05
        dy[-1] = 0.05
        dz = 0.1 * np.ones_like(z)
        dz[0] = 0.05
        dz[-1] = 0.05
        dx_grid, dy_grid, dz_grid = np.meshgrid(dx, dy, dz, indexing='ij')
        correct_volume = dx_grid * dy_grid * dz_grid
        grid_object = gr.CartesianGrid((x, y, z))
        calculated_grid = grid_object.grid
        calculated_volume = grid_object.volume
        for correct, calculated in zip(correct_grid, calculated_grid):
            self.assertTrue(np.array_equal(correct, calculated))
        self.assertTrue(np.array_equal(np.round(calculated_volume, 7), np.round(correct_volume, 7)))


class TestSphericalGrid(TestCase):
    def test_spherical_grid_init(self):
        r_array = (1., 0.5, 0.0)
        theta_array = (0.0, pi / 2)
        phi_array = (0.0, pi / 2)
        correct_grid = sp.spherical_to_cartesian_mesh((r_array, theta_array, phi_array))
        r = correct_grid[0]
        theta = correct_grid[1]
        diffs = (np.ones_like(r), np.ones_like(r), np.ones_like(r))
        correct_volume = sp.calculate_spherical_volume_element(r, theta, diffs)
        grid_object = gr.SphericalGrid((r_array, theta_array, phi_array))
        calculated_grid = grid_object.grid
        calculated_volume = grid_object.volume
        for correct, calculated in zip(correct_grid, calculated_grid):
            self.assertTrue(np.array_equal(correct, calculated))
        self.assertTrue(np.array_equal(np.round(calculated_volume, 7), np.round(correct_volume, 7)))



if __name__ == '__main__':
    unittest.main()
