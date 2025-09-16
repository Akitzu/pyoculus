from pyoculus.fields import AxisymmetricCylindricalGridField
from pyoculus.maps import CylindricalBfieldSection
from pyoculus.solvers import PoincarePlot, FixedPoint
import numpy as np
from matplotlib import pyplot as plt
import logging
logging.basicConfig(level=logging.DEBUG)

DoubletField = AxisymmetricCylindricalGridField.from_matlab_file('./struct_for_chris.mat', with_perturbation=True)

section = CylindricalBfieldSection(DoubletField, R0=0.88, Z0=0, rtol=1e-10)


top_o = FixedPoint(section)
guess = [0.88, 0.4]
point = [0.88, 0., 0.4]
top_o.find(1, guess)

section.f(1, guess)

top = np.array([0.88, .4])
middle = np.array([0.88, 0])
out = np.array([0.98, 0])
bottom = np.array([0.88, -.4])

pplot = PoincarePlot.with_segments(section, [top, bottom, middle, out], [100, 50], connected=False)
pplot.compute(npts=3000, nprocess=1)
pplot.plot()
np.save('doublet_poincare_hits', pplot._hits)

plt.show()

#points = [np.array([0.88, 0, z]) for z in np.linspace(.4, -.4, 1000)]
#perturbation_ratio = [np.linalg.norm(DoubletField.pertfun(xx)[::2])/np.linalg.norm(DoubletField.B_axi(xx)[::2]) for xx in points]


