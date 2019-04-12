@ECHO OFF
SETLOCAL

SET _source="C:\ALL_DATA\URGummitha\Desktop\PreMod-M7"

SET _dest="G:\PROJECTS\CAEP1073 - EIS PC-12 Radar Installation\wip\FEM\PreMod- M7\Optistruct_Runs"

SET _what=/COPY:dt /MIR
:: /COPYALL :: COPY ALL file info
:: /B :: copy files in Backup mode. 
:: /MIR :: MIRror a directory tree 

SET _options=/R:0 /W:0 /NFL /NDL /mon:1 /mot:1
:: /R:n :: number of Retries
:: /W:n :: Wait time between retries
:: /LOG :: Output log file
:: /NFL :: No file logging
:: /NDL :: No dir logging

ROBOCOPY %_source% %_dest% %_what% %_options%
pause