import unittest
import QuDiPy.coordinates.cartesian
from unittest import TestCase
import numpy as np


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


if __name__ == '__main__':
    unittest.main()
