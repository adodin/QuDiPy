import unittest
from unittest import TestCase
import QuDiPy.util.grids as gr
import QuDiPy.util.spherical as sp
import numpy as np
from numpy import pi


class TestCalculate_differentials(TestCase):
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
        calculated_diffs = gr.calculate_differentials((x, y, z))
        for correct, calc in zip(correct_differentials, calculated_diffs):
            rounded_calc = np.round(calc, 7)
            self.assertTrue(np.array_equal(rounded_calc, correct))

    def test_calculate_nonuniformly_spaced_differentials(self):
        x = (0., 0.1, 0.3, 0.9)
        dx = (0.05, 0.15, 0.4, 0.3)
        correct_differentials = (dx,)
        calculated_diffs = gr.calculate_differentials((x,))
        for correct, calc in zip(correct_differentials, calculated_diffs):
            rounded_calc = np.round(calc, 7)
            self.assertTrue(np.array_equal(rounded_calc, correct))


class TestCalculate_cartesian_volume_element(TestCase):
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
        calculated_volume = gr.calculate_cartesian_volume_element((x, y, z))
        self.assertTrue(np.array_equal(np.round(calculated_volume, 7), np.round(correct_volume, 7)))


class TestCalculate_cartesian_gradient(TestCase):
    def test_calculate_cartesian_gradient_linear(self):
        x = np.linspace(-1., 1., 11)
        y = np.linspace(-1., 1., 21)
        z = np.linspace(0., 1., 11)
        x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing='ij')
        funct = z_grid
        correct_gradient = (np.zeros_like(x_grid), np.zeros_like(y_grid), np.ones_like(z_grid))
        calculated_gradient = gr.calculate_cartesian_gradient(funct, (x, y, z))
        for correct, calculated in zip(correct_gradient, calculated_gradient):
            self.assertTrue(np.array_equal(np.round(calculated, 7), correct))

    def test_calculate_cartesian_gradient_xy(self):
        x = np.linspace(-1., 1., 11)
        y = np.linspace(-1., 1., 21)
        z = np.linspace(0., 1., 11)
        x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing='ij')
        funct = x_grid * y_grid
        correct_gradient = (y_grid, x_grid, np.zeros_like(z_grid))
        calculated_gradient = gr.calculate_cartesian_gradient(funct, (x, y, z))
        for correct, calculated in zip(correct_gradient, calculated_gradient):
            self.assertTrue(np.array_equal(np.round(calculated, 7), np.round(correct, 7)))

    def test_calculate_cartesian_gradient_quadratic(self):
        x = np.linspace(-1., 1., 11)
        y = np.linspace(-1., 1., 21)
        z = np.linspace(0., 1., 11)
        x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing='ij')
        funct = z_grid ** 2
        correct_gradient = (np.zeros_like(x_grid), np.zeros_like(y_grid), 2 * z_grid)
        calculated_gradient = gr.calculate_cartesian_gradient(funct, (x, y, z))
        for correct, calculated in zip(correct_gradient, calculated_gradient):
            self.assertTrue(np.array_equal(np.round(calculated, 7), np.round(correct, 7)))

    ''' # Slow Test That should be run before push only
    def test_calculate_cartesian_gradient_gausian(self):
        x = np.linspace(-1., 1., 11)
        y = np.linspace(-1., 1., 21)
        z = np.linspace(0., 1., 100000)
        x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing='ij')
        funct = np.exp(-z_grid ** 2)
        correct_gradient = (np.zeros_like(x_grid), np.zeros_like(y_grid), -2 * z_grid * funct)
        calculated_gradient = gr.calculate_cartesian_gradient(funct, (x, y, z))
        for correct, calculated in zip(correct_gradient, calculated_gradient):
            test = np.round(calculated, 4) == np.round(correct, 4)
            sum_test = np.sum(test)
            self.assertTrue(np.array_equal(np.round(calculated, 4), np.round(correct, 4))) # Up to Numerical Error'''


class TestCalculate_cartesian_divergence(TestCase):
    def test_calculate_cartesian_divergence_whirlpool(self):
        x = np.linspace(-1., 1., 11)
        y = np.linspace(-1., 1., 21)
        z = np.linspace(0., 1., 11)
        x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing='ij')
        funct = (y_grid, x_grid, np.zeros_like(z_grid))
        correct_divergence = np.zeros_like(x_grid)
        calculated_divergence = gr.calculate_cartesian_divergence(funct, (x, y, z))
        test = np.round(calculated_divergence, 7) == correct_divergence
        self.assertTrue(np.array_equal(np.round(calculated_divergence, 7), correct_divergence))

    def test_calculate_cartesian_divergence_radial(self):
        x = np.linspace(-1., 1., 11)
        y = np.linspace(-1., 1., 21)
        z = np.linspace(0., 1., 11)
        x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing='ij')
        funct = gr.calculate_cartesian_gradient(.5*(x_grid ** 2 + y_grid ** 2 + z_grid**2), (x, y, z))
        correct_divergence = 3*np.ones_like(x_grid)
        calculated_divergence = gr.calculate_cartesian_divergence(funct, (x, y, z))
        test = np.round(calculated_divergence, 7) == correct_divergence
        self.assertTrue(np.array_equal(np.round(calculated_divergence, 7), correct_divergence))

    def test_calculate_cartesian_gradient_quadratic(self):
        x = np.linspace(-1., 1., 11)
        y = np.linspace(-1., 1., 21)
        z = np.linspace(0., 1., 11)
        x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing='ij')
        funct = z_grid ** 2
        correct_gradient = (np.zeros_like(x_grid), np.zeros_like(y_grid), 2 * z_grid)
        calculated_gradient = gr.calculate_cartesian_gradient(funct, (x, y, z))
        for correct, calculated in zip(correct_gradient, calculated_gradient):
            self.assertTrue(np.array_equal(np.round(calculated, 7), np.round(correct, 7)))



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
        correct_grid = sp.spherical_to_cartesian_grid((r_array, theta_array, phi_array))
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
