mkdir working
cd working
git clone https://github.com/alexc17/syntheticpcfg.git
git clone https://github.com/alexc17/testpcfg.git
git clone https://github.com/alexc17/locallearner.git
cd testpcfg/scripts
./generate_data.sh
mv ../data ../../locallearner/
cd ../../locallearner
mkdir bin
cd lib
make io
mv io ../bin
cd ../scripts
nohup make -j 50 json &

