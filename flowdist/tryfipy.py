"""This takes commands directly from the Stokes Flow example
from FiPy in an attempt to adapt the process for my own
purposes. The intention at first to to perform the same
Stokes Flow simulation without much extra stuff going on
and to learn the syntax."""

from fipy.meshes.grid2D import Grid2D
from fipy.variables import CellVariable, FaceVariable, faceGradVariable
from fipy.terms import DiffusionTerm
from fipy.viewers import Viewer

length = 1.0
n = 50
dl = length/n
viscosity = 1
u = 1.0
pressureRelaxation = 0.825
velocityRelaxation = 0.55
sweeps = 500
tolerance = 10**(-3)
mesh = Grid2D(nx=n, ny=n, dx=dl, dy=dl)
pressure = CellVariable(mesh=mesh, name='pressure')
pressure_corr = CellVariable(mesh=mesh)
velocity_x = CellVariable(mesh=mesh, name='X velocity')
velocity_y = CellVariable(mesh=mesh, name='Y velocity')

velocity = FaceVariable(mesh=mesh, rank=1)

velocity_x_eq = DiffusionTerm(coeff=viscosity) - pressure.grad.dot([1.0, 0.0])
velocity_y_eq = DiffusionTerm(coeff=viscosity) - pressure.grad.dot([0.0, 1.0])

ap = CellVariable(mesh=mesh, value=1.0)
coefficient = 1.0/ap.arithmeticFaceValue*mesh._faceAreas*mesh._cellDistances
pressure_corr_eq = DiffusionTerm(coeff=coefficient) - velocity.divergence

volume = CellVariable(mesh=mesh, value=mesh.cellVolumes, name='Volume')
control_volume = volume.arithmeticFaceValue

velocity_x.constrain(0.0, mesh.facesRight | mesh.facesLeft | mesh.facesBottom)
velocity_x.constrain(u, mesh.facesTop)
velocity_y.constrain(0.0, mesh.exteriorFaces)
x, y = mesh.faceCenters
pressure_corr.constrain(0.0, mesh.facesLeft & (y < dl))

viewer = Viewer(vars=(pressure, velocity_x, velocity_y),
                xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, colorbar=True)

res_max = 2.0*tolerance
iteration = 1
while res_max > tolerance:
    velocity_x_eq.cacheMatrix()
    res_x = velocity_x_eq.sweep(var=velocity_x, underRelaxation=velocityRelaxation)
    matx = velocity_x_eq.matrix
    res_y = velocity_y_eq.sweep(var=velocity_y, underRelaxation=velocityRelaxation)
    ap[:] = -matx.takeDiagonal()

    pressure_grad = pressure.grad
    facepressure_grad = faceGradVariable._FaceGradVariable(pressure)

    velocity[0] = velocity_x.arithmeticFaceValue + \
        control_volume / ap.arithmeticFaceValue * \
        (pressure_grad[0].arithmeticFaceValue - facepressure_grad[0])

    velocity[1] = velocity_y.arithmeticFaceValue + \
        control_volume / ap.arithmeticFaceValue * \
        (pressure_grad[1].arithmeticFaceValue - facepressure_grad[1])

    # velocity[..., mesh.exteriorFaces.value] = 0.0
    velocity[:, mesh.exteriorFaces.value] = 0.0
    velocity[0, mesh.facesTop.value] = u

    pressure_corr_eq.cacheRHSvector()
    res_p = pressure_corr_eq.sweep(var=pressure_corr)
    res_max = max(res_x, res_y, res_p)
    rhs = pressure_corr_eq.RHSvector

    pressure.setValue(pressure + pressureRelaxation * pressure_corr)
    velocity_x.setValue(velocity_x - pressure_corr.grad[0] / ap * mesh.cellVolumes)
    velocity_y.setValue(velocity_y - pressure_corr.grad[1] / ap * mesh.cellVolumes)

    if iteration % 10 == 0:
        print 'sweep: ', iteration, ', x residual: ', res_x, \
                                ', y residual: ', res_y, \
                                ', p residual: ', res_p, \
                                ', continuity:', max(abs(rhs))

        viewer.plot()

    iteration += 1
