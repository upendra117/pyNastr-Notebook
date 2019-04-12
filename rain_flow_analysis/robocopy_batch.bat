@ECHO OFF
SETLOCAL

SET _source="C:\ALL_DATA\URGummitha\Desktop\Post_Mod-2\Run_M2.52"

SET _dest="C:\ALL_DATA\URGummitha\Desktop\Post_Mod-2\Cradle_Run\Run_M2.54_d1"

SET _what=/COPY:dt /XF *.op2 *.xdb *.f06 *.f04 *.h5 *.bdf.log *.bdf.rej /MIR
:: /COPYALL :: COPY ALL file info
:: /B :: copy files in Backup mode. 
:: /MIR :: MIRror a directory tree 

SET _options=/R:0 /W:0 /NFL /NDL /mon:1 /mot:1
:: /R:n :: number of Retries
:: /W:n :: Wait time between retries
:: /LOG :: Output log file
:: /NFL :: No file logging
:: /NDL :: No dir logging
:: /XF *.op2 *.xdb *.f06 *.f04 *.h5 :: Exclude file types

ROBOCOPY %_source% %_dest% %_what% %_options%

pause