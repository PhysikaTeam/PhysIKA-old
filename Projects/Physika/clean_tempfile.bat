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
del /F /S /Q *.aps *.idb *.ncp *.obj *.pch *.sbr *.tmp *.pdb *.bsc *.ilk *.res *.ncb *.opt *.suo *.manifest *.dep *.tlog *.log *.lastbuildstate *.rc *.exe
rd bin
rd Debug
exit 