#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import sys
import os


fn = sys.argv[1]
plt.plot(np.load(fn).transpose())
plt.legend(['Real', 'Length min', 'IAT min', 'Length max', 'IAT max'])
plt.xlabel('Sequence index')
plt.ylabel('Prediction')
#plt.savefig('%s.pdf' % os.path.splitext(fn)[0])
plt.show()

