from __future__ import division, print_function

ProgramShort = 'ILT'
ProgramName ='1D and 2D Inverse Laplace Transform'
VersionName = 'Development version'
version = 1.0
revision = 2
rev_date = '17-Nov-2018'

def report():
    "prints version name when program starts"
    return( '''
=================================
{0}
{1}
=================================
Version : {2}
Date : {3}
Revision Id : {4}
=================================
'''.format(ProgramShort,ProgramName, version, rev_date, revision))
print(report())
