@echo off
echo ----------------------------------------------------
echo By MoreWindows (http://blog.csdn.net/MoreWindows)
echo Press any key to delete all files with ending:
echo  *.aps *.idb *.ncp *.obj *.pch *.tmp *.sbr
echo  *.tmp *.pdb *.bsc *.ilk *.res *.ncb *.opt 
echo  *.suo *.manifest  *.dep
echo  *.tlog *.log *.lastbuildstate *.rc
echo  *.exe
echo There are Visual C++ and Visual Studio junk
echo ----------------------------------------------------
del /F /S /Q *.lib
cd include
RD /S/Q Physika_Core
RD /S/Q Physika_Dependency
RD /S/Q Physika_Dynamics
RD /S/Q Physika_Geometry
RD /S/Q Physika_GUI
RD /S/Q Physika_IO
RD /S/Q Physika_Render
exit 