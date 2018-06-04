"""This takes many components from the Stokes flow example and adapts it to a 2D channel with a pressure boundary
condition"""

from fipy.meshes import nonUniformGrid2D
from fipy.variables import CellVariable, FaceVariable, faceGradVariable
from fipy.terms import DiffusionTerm
from fipy.viewers import Viewer

length = 1.0
channel_height = length/10
channel_depth = length/20
nx = 20
ny = 20
nz = 20
dx = length/nx
dy = channel_height/ny
dz = channel_depth/nz
viscosity = 100
pressure_inlet = 1.0
pressureRelaxation = 0.8
velocityRelaxation = 0.5
tolerance = 10**(-6)
mesh = nonUniformGrid2D.NonUniformGrid2D(nx=nx, ny=ny, dx=dx, dy=dy)
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

# Boundary Conditions
velocity_x.constrain(0.0, mesh.facesTop | mesh.facesBottom)
velocity_y.constrain(0.0, mesh.facesTop | mesh.facesBottom)

x, y = mesh.faceCenters

pressure_corr.constrain(0.0, mesh.facesLeft & (x < dx))
pressure_corr.constrain(0.0, mesh.facesRight & (x > length - dx))
pressure.constrain(pressure_inlet, mesh.facesLeft & (x < dx))
pressure.constrain(0.0, mesh.facesRight & (x > length - dx))

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

    velocity[0] = velocity_x.arithmeticFaceValue + control_volume / ap.arithmeticFaceValue * \
        (pressure_grad[0].arithmeticFaceValue - facepressure_grad[0])

    velocity[1] = velocity_y.arithmeticFaceValue + control_volume / ap.arithmeticFaceValue * \
        (pressure_grad[1].arithmeticFaceValue - facepressure_grad[1])

    velocity[:, mesh.facesTop.value] = 0.0
    velocity[:, mesh.facesBottom.value] = 0.0

    pressure_corr_eq.cacheRHSvector()
    res_p = pressure_corr_eq.sweep(var=pressure_corr)
    res_max = max(res_x, res_y, res_p)
    rhs = pressure_corr_eq.RHSvector

    pressure_corr.arithmeticFaceValue.setValue(0.0, where=(x <= dx))
    pressure_corr.arithmeticFaceValue.setValue(0.0, where=(x > length - dx))

    pressure.setValue(pressure + pressureRelaxation * pressure_corr)
    # pressure.setValue(pressure_inlet, where=(x < dx))
    # pressure.setValue(0.0, where=x >= length - dx)
    pressure.arithmeticFaceValue.setValue(pressure_inlet, where=(x < 0.5*dx))
    pressure.arithmeticFaceValue.setValue(0.0, where=(x > length - 0.5*dx))

    velocity_x.setValue(velocity_x - pressure_corr.grad[0] / ap * mesh.cellVolumes)
    velocity_y.setValue(velocity_y - pressure_corr.grad[1] / ap * mesh.cellVolumes)

    if iteration % 10 == 0:
        print 'Iteration: ', iteration, ', x residual: ', res_x, \
                                ', y residual: ', res_y, \
                                ', p residual: ', res_p, \
                                ', continuity:', max(abs(rhs))

        viewer.plot()

    iteration += 1
