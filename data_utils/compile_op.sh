cd randlanet_utils/nearest_neighbors
python setup.py install --home="."
cd ../../

cd randlanet_utils/cpp_wrappers
sh compile_wrappers.sh
cd ../../../