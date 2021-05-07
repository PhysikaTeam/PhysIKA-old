@echo off

cd ../build/bin/

cls

echo.
echo.
echo ######################请选择要执行的操作######################
echo ----------------------1、流固耦合----------------------
echo ----------------------2、碰撞----------------------
echo ----------------------3、多种模型表示方式对比----------------------
echo ----------------------4、多种计算方法对比----------------------
echo ----------------------5、布料----------------------
echo.
echo.
echo 请选择要执行的操作
set /p num=
echo %num%

if "%num%" == "1" (
   .\App_SFIHybrid.exe
)
if "%num%" == "2" (
   .\App_CollisionHybridSix.exe
)
if "%num%" == "3" (
   .\App_CollisionHybridSixTwo.exe
)
if "%num%" == "4" (
   .\App_CollisionHybridTwo.exe
)
if "%num%" == "5" (
   .\App_Cloth.exe
)


cd ../../Run
