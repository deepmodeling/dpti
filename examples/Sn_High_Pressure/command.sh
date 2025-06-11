# NPT
dpti equi gen npt.json -e npt-xy -t 200 -p 20000  -o NPT_sim/
# cd NPT_sim/new_job/
dpti equi extract ./  -o npt_avg.lmp

# NVT
dpti equi gen equi_settings.json --ensemble nvt -t 200 -p 20000 --conf-npt ./NPT_sim/new_job/ -o NVT_sim/
# cd NVT_sim/new_job/
dpti equi extract ./  -o nvt_last_dump.lmp

# HTI
dpti hti gen hti.json -s three-step -o HTI_sim/
# dpti hti_water gen hti_water.json -o HTI_water/
# dpti hti_ice gen hti_ice.json  -s three-step -o HTI_ice/
dpti hti compute ./new_job/  -t gibbs --npt ../NPT_sim/new_job/

# TI
dpti ti gen ti_settings.json -o TI_sim/
dpti ti compute ./TI_sim/new_job/  --hti ../HTI_sim/new_job/



