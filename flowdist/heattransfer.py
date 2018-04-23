import ht.conv_internal
import CoolProp.CoolProp.PropsSI as cp
import math

# for now, heat transfer correlations are chosen based on whether the
# flow is laminar or turbulent
# the channels are assumed to be circular with a constant heat flux
# this will be updated as transience is introduced into the problem
# it is anticipated that correlations will not be used later since
# they do not capture transience well
# fully developed flow is also assumed

def choosehtcorr(reynolds, prandtl):
    if reynolds >= 1 or reynolds <= 10000:
        nusselt = ht.conv_internal.laminar_Q_const()

    elif reynolds > 10000 and prandtl >= 0.6 and prandtl >= 160:
        nusselt = ht.conv_internal.turbulent_Dittus_Boelter(reynolds, prandtl, heating = True, revised = False)

    else:
        return 'No Nusselt correlation available for this Prandtl or Reynolds number' + 'Re = ' + str(reynolds) + ', Pr = ' str(prandtl)

    return nusselt


def temperature_surface(nusselt, conductivity, diameter, temperature_bulk, heatflux):
    htcoeff_convective = nusselt*conductivity/diameter
    return (heatflux/htcoeff_convective) + temperature_bulk


def temperatured_out(fluid, flowrate_mass, diameter, heatflux, temperature_bulk_in, pressure):
    enthalpy_in = cp('h', 'T', temperature_bulk_in, 'P', pressure, fluid)
    area_heattransfer = math.pi*diameter
    massflux = flowrate_mass/area_heattransfer
    enthalpy_out = (heatflux/massflux) + enthalpy_in
    return cp('T', 'h', enthalpy_out, 'P', pressure, fluid)