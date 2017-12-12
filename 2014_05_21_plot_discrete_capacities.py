from __future__ import division
from pylab import *
from scipy import interpolate, stats, optimize
import itertools

from myCodeCopy import JessePlot, CapacityTools, blahut


def loadPulseData(filename, suffix = ''):
    """Given a pulse text file, puts the data into the global namespace
    """
    data = np.genfromtxt(filename+'.txt', skip_header=3, names=True,
                   dtype='i8,f8,S5,f8,f8,f8,f8,f8,f8')
    print "Importing...\n"
    for key in data.dtype.fields.keys():
        name = key + suffix
        print name
        globals()[name] = data[key]


#==============================================================================
#%% Load and preformat the data
#Derive 2-D Density with Linear Interpolation of Gaussian KDE
#==============================================================================
loadPulseData('./../data/FD12W_2_PS_4')
set_ = [Type=='Set']
first_run = 200
last_run = 580
pts_per_run = 40
Rarray = R[set_][first_run*pts_per_run:last_run*pts_per_run]
Varray = V[set_][first_run*pts_per_run:last_run*pts_per_run]
Rarray = log10(Rarray)


# Gaussian KDE
V_list =  Varray[0:pts_per_run]

Pyx_func = []
for v in V_list:  
    data = Rarray[ Varray == v ]    
    print v, sum(Varray == v)
    Pyx_func.append(stats.gaussian_kde(data, bw_method='scott' )) #scott, silvermann, scalar
Pyx_func=CapacityTools.FunctionList(Pyx_func)


# Interpolate
nx=2000
ny=2000

m1 = Varray
m2 = Rarray

xmin = m1.min()
xmax = m1.max()
ymin = m2.min()
ymax = m2.max()

x = linspace(xmin, xmax, nx)
y = linspace(ymin, ymax, ny)

# Bivariate Spline
Pyx = np.atleast_2d(Pyx_func(y))
Pyx_interp = interpolate.RectBivariateSpline( V_list, y, Pyx, kx=3, ky=3, s=0)
Pyx = np.rot90(Pyx_interp(x,y))

Pyx = Pyx / np.sum(Pyx, axis=0)

#==============================================================================
# %% PLOTTING
#
# Load Simulation Results
#==============================================================================
r = load('./../npz/C_equal_optimized.npz')['data'].item()
C_equal= (r['C_equal'])
C_equal_bounds= (r['C_equal_bounds'])


r = load('./../npz/C_nonequal_optimized.npz')['data'].item()
C_nonequal = r['C_nonequal']
x_in_nonequal = r['x_in']
Px_in_nonequal = r['Px_in']
y_out_nonequal = r['y_out']


r = load('./../npz/basinhopping.npz')['data'].item()
C_out, y_div, Px_out, C_in, x_in, Px_in = (r['C_out'], 
                                           r['y_div'], 
                                           r['Px_out'],
                                           r['C_in'], 
                                           r['x_in'], 
                                           r['Px_in'])

C_in = array(C_in)
C_out = array(C_out)

nbits = 8


#==============================================================================
# %% Plot Capacity Matrix
#==============================================================================
reload(JessePlot)

JessePlot.rc('word_3')
C_equal[0,:]=1
C_equal[:,0]=1

imshow(rot90(C_equal), extent=[0.5,8.5,0.5,8.5], interpolation='None', cmap=cm.Reds)
bits = arange(nbits)+1

xticks(bits)
yticks(bits[::-1])

cb = colorbar()
cb.set_label('Capacity (Bits)') #, rotation=270)
cb.set_ticks([1,1.5,2,2.5])
cb.ax.get_yaxis().labelpad = 3


xlabel('DAC (Input) Bits')
ylabel('ADC (Output) Bits')
tick_params(axis='both', top='off', right='off')
tight_layout()
show()
JessePlot.saveFig('./../figs/CvsNbits_matrix.png')



##%%
#JessePlot.rc('ppt_1')
#for i in range(4):
#    plot(C_equal_bounds.diagonal()[i,:])
#    show()
#
#

#==============================================================================
# %% Plot Capacity vs. # of states
#==============================================================================
reload(JessePlot)

JessePlot.rc('word_3')

nin = arange(len(C_in))
nout = arange(len(C_out))+1                                          
bits = arange(nbits)+1

bits_in = r_[log2(nin[r_[2,4,5,8,12]]), arange(4,9)] #16
C_in_plot = r_[ C_in[r_[2,4,5,8,12]], C_in[12]*ones(5) ]

#bits_out = r_[log2(nout[1:8]), log2(nout[15]), log2(nout[31]), log2(nout[63]), arange(7,9)]
bits_out = bits
C_out_plot = r_[C_out[1], C_nonequal.diagonal()[1:3], C_out[15], C_out[31], C_out[63], C_equal.diagonal()[6:]]

plot([0,100],[2.68,2.68], 'k--')
plot(bits, bits, 'k:', linewidth=2)

plot(bits_in, C_in_plot, '-o', label='Analog Output')
plot(bits_out, C_out_plot, '-^r', label='Optimal Spacing')
plot(bits, C_equal.diagonal() , 'k-s', label='Equal Spacing')



text(2.1, 3.0, 'Ideal', rotation=70)
text(3.3, 2.8, 'Analog = 2.68 Bits')
legend(fontsize=8,loc='lower right')


xlabel('ADC/DAC Bits')
ylabel('Capacity (Bits)')
xlim(1, 8)
ylim(1, 3.2)

ax = gca()
ax.xaxis.labelpad = 1
ax.xaxis.pad = 0
ax.tick_params(top=None, right=None)
tight_layout(pad=0.3)


show()
JessePlot.saveFig('./../figs/CvsNbits.png')




#==============================================================================
# %% Plot Px's
#==============================================================================
JessePlot.rc('word_3')
for i, j in enumerate(r_[2:15]):        
    nstates = j

    if j > 12:
        j=12
    scatter( x_in[j],nstates*ones(len(x_in[j])), s=8,
            c=Px_in[j], vmin=0, vmax=0.5, cmap=cm.Blues)


ylim(0, 16)
xlim(0.5, 3.5)
yticks(arange(2, 16, 2))
xticks([1,2,3])

ylabel('# Input States')
#gca().set_xticks(arange(8,0,-1))
#gca().set_yticks(arange(-10,15,5))

xlabel('V$_{WL}$ (V)')

cb = colorbar()
cb.set_label('P(V$_{WL}$)')
cb.set_ticks(arange(0, 0.6, 0.1))
cb.ax.get_yaxis().labelpad = 5

gca().tick_params(right=None, top=None)


tight_layout()
show()
JessePlot.saveFig('./../figs/PxVsNstates.png')


#==============================================================================
#%% Plot example Discretization 
#==============================================================================
x_ana = x_in[4]
Px_ana = Px_in[4]

eq = C_equal_bounds[1,1]
xeq = linspace(eq[0], eq[1], 4)
yeq_bound = linspace(eq[2], eq[3], 5)
yeq = yeq_bound[1:-1]


Pyx_sub_eq, x_sub_eq, y_sub_eq = blahut.quantize(Pyx, x, y, xeq, yeq)
Ceq, Px_eq = blahut.blahut_arimoto(Pyx_sub_eq, 
                              tolerance=1e-7, 
                              iterations = 100)

xneq = x_in_nonequal[1][1]
Pxneq = x_in_nonequal[1][1]
yneq = y_out_nonequal[1][1]

Pyx_sub_neq, x_sub_neq, y_sub_neq = blahut.quantize(Pyx, x, y, xneq, yneq)
Cneq, Px_neq = blahut.blahut_arimoto(Pyx_sub_neq, 
                              tolerance=1e-7, 
                              iterations = 100)


#%% Plot Py
JessePlot.rc('word_3')

PRmax = 0.008
PVmax = 0.35


subplot(322)
Py = zeros(Pyx.shape[0])
for i in arange(4):
    ind = blahut.find_closest(x, x_sub_eq[i])   
    color = ['r', 'orange', 'green', 'blue'][i]    
    
    PyxPx = dot(Pyx[:,ind], Px_eq[i])[::-1].flatten()
    plot(y, PyxPx, '-', color=color )
    fill_between(y, PyxPx, color=color, alpha=0.1)
    
    Py += PyxPx

for yeq_ in y_sub_eq:
    line, = plot([yeq_, yeq_], [0,10], 'k--', linewidth=1)
    line.set_dashes([4,1]) 


#plot(y, Py, 'k-')
ylim(0, PRmax)
xlim(4, 7.0)
tick_params(left=None, labelleft=None, right=None, bottom=None, labelbottom=None, top=None)




subplot(324)
Py = zeros(Pyx.shape[0])
for i in arange(4):
    ind = blahut.find_closest(x, x_sub_neq[i])   
    color = ['r', 'orange', 'green', 'blue'][i]    
    
    PyxPx = dot(Pyx[:,ind], Px_neq[i])[::-1].flatten()
    plot(y, PyxPx, '-', color=color )
    fill_between(y, PyxPx, color=color, alpha=0.1)
    
    Py += PyxPx

for yeq_ in y_sub_neq:
    line, = plot([yeq_, yeq_], [0,10], 'k--', linewidth=1)
    line.set_dashes([4, 1]) 

#plot(y, Py, 'k-')
ylim(0, PRmax)
xlim(4, 7.0)
tick_params(left=None, labelleft=None, right=None, bottom=None, labelbottom=None, top=None)
ylabel('P(R)', fontsize=8)



subplot(326)
Py = zeros(Pyx.shape[0])
for i in arange(4):
#    ind = blahut.find_closest(x, x_ana[i])   
#    color = ['orange', 'r', 'b', 'g'][i]    
#    
#    PyxPx = dot(Pyx[:,ind], Px_ana[i])[::-1].flatten()
#    plot(y, PyxPx, '-', color=color )
#    fill_between(y, PyxPx, color=color, alpha=0.1)
#
#    Py += PyxPx


    ind = blahut.find_closest(x, x_sub_neq[i])   
    color = ['r', 'orange', 'green', 'blue'][i]    
    
    PyxPx = dot(Pyx[:,ind], Px_neq[i])[::-1].flatten()
    plot(y, PyxPx, '-', color=color )
    fill_between(y, PyxPx, color=color, alpha=0.1)
    
    Py += PyxPx


#plot(y, Py, 'k-')
ylim(0, PRmax)
xlim(4, 7.0)
tick_params(left=None, labelleft=None, right=None, bottom=None, labelbottom=None, top=None)
#xticks([4, 5, 6, 7])
xlabel('R', fontsize=8)



#Px
subplot(321)
for i in arange(4):
    Px = zeros(size(x))
    ind = blahut.find_closest(x, x_sub_eq[i])   
    color = ['r', 'orange', 'green', 'blue'][i] 
    Px[ind] = Px_eq[i]
    plot(x, Px, '-', color=color )
    
ylim(0, PVmax)
xlim(0.5, 3.5)
tick_params(left=None, labelleft=None, right=None, bottom=None, labelbottom=None, top=None)
ylabel('Equal\n', fontsize=8)


subplot(323)
for i in arange(4):
    Px = zeros(size(x))
    ind = blahut.find_closest(x, x_sub_neq[i])   
    color = ['r', 'orange', 'green', 'blue'][i] 
    Px[ind] = Px_neq[i]
    plot(x, Px, '-', color=color )
    
ylim(0, PVmax)
xlim(0.5, 3.5)
tick_params(left=None, labelleft=None, right=None, bottom=None, labelbottom=None, top=None)
ylabel('Optimal\nP(V$_{WL}$)', fontsize=8)


subplot(325)
for i in arange(4):
#    Px = zeros(size(x))
#    ind = blahut.find_closest(x, x_ana[i])   
#    color = ['orange', 'r', 'b', 'g'][i] 
#    Px[ind] = Px_ana[i]
#    plot(x, Px, '-', color=color )
    Px = zeros(size(x))
    ind = blahut.find_closest(x, x_sub_neq[i])   
    color = ['r', 'orange', 'green', 'blue'][i] 
    Px[ind] = Px_neq[i]
    plot(x, Px, '-', color=color )

    
ylim(0, PVmax)
xlim(0.5, 3.5)
tick_params(left=None, labelleft=None, right=None, bottom=None, labelbottom=None, top=None)
ylabel('Analog\n', fontsize=8)
#xticks([1,2,3])
xlabel('V$_{WL}$', fontsize=8)

#yticks([1,2,3])


tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
show()
JessePlot.saveFig('./../figs/P_R_quantize_example.png')



