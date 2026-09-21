#include "core/simulator.hpp"
#include "core/conversion.hpp"
#include <yaml-cpp/yaml.h>
#include <opencv2/imgcodecs.hpp>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <future>
#include <stdexcept>
#include <cstdlib>
#include <iostream>

void check(bool ok) { if (!ok) throw std::runtime_error("simulator protocol assertion failed"); }
void read_all(int fd, void* p, size_t n) {
    auto* out=static_cast<char*>(p);
    while(n) { auto r=recv(fd,out,n,0); check(r>0); out+=r; n-=r; }
}
YAML::Node read_json(int fd) {
    uint32_t n; read_all(fd,&n,4); n=ntohl(n); check(n<4096);
    std::string s(n,'\0'); read_all(fd,s.data(),n); return YAML::Load(s);
}
int main() try {
    setenv("DAEDALUS_BRIDGE_TOKEN","test-token-not-for-deployment",1);
    int listener=socket(AF_INET,SOCK_STREAM,0); check(listener>=0);
    sockaddr_in address{}; address.sin_family=AF_INET; address.sin_addr.s_addr=htonl(INADDR_LOOPBACK);
    check(bind(listener,reinterpret_cast<sockaddr*>(&address),sizeof(address))==0);
    check(listen(listener,1)==0); socklen_t len=sizeof(address);
    check(getsockname(listener,reinterpret_cast<sockaddr*>(&address),&len)==0);
    auto server=std::async(std::launch::async,[&] {
        int fd=accept(listener,nullptr,nullptr); check(fd>=0);
        timeval timeout{5,0}; setsockopt(fd,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout));
        auto hello=read_json(fd); check(hello["version"].as<int>()==1);
        check(hello["token"].as<std::string>()=="test-token-not-for-deployment");
        // Optical->world quaternion at a level, forward-facing FLU camera.
        std::string meta=R"({"version":1,"frame_seq":7,"capture_unix_ns":1000000000,"width":32,"height":24,"fx":30,"fy":30,"cx":16,"cy":12,"yaw_deg":0,"pitch_deg":0,"camera_position":[11,22,33],"muzzle_position":[10,20,30],"camera_orientation_xyzw":[-0.5,0.5,-0.5,0.5],"chassis_yaw_deg":45,"bore_pitch_deg":25,"auto_aim_enabled":true,"sim_fire_allowed":true,"simulated_shots":0,"simulated_hits":0})";
        std::vector<uchar> jpeg; cv::imencode(".jpg",cv::Mat::zeros(24,32,CV_8UC3),jpeg);
        uint32_t sizes[]{htonl(meta.size()),htonl(jpeg.size())};
        std::string packet(reinterpret_cast<char*>(sizes),8);packet+=meta;packet.append(reinterpret_cast<char*>(jpeg.data()),jpeg.size());
        // Deliberately fragmented packets must be reconstructed correctly.
        for(char c:packet) check(send(fd,&c,1,MSG_NOSIGNAL)==1);
        auto cmd=read_json(fd);
        check(cmd["frame_seq"].as<int>()==7 && cmd["source_frame_seq"].as<int>()==7);
        check(cmd["target_valid"].as<bool>() && cmd["fire"].as<bool>());
        check(std::abs(cmd["yaw_deg"].as<double>()-10)<1e-6);
        check(std::abs(cmd["pitch_deg"].as<double>()-5)<1e-6);
        uint32_t invalid[]{htonl(5000),htonl(1)};check(send(fd,invalid,8,MSG_NOSIGNAL)==8);
        close(fd);
    });
    {
        rmcs::SimulatorClient client("127.0.0.1:"+std::to_string(ntohs(address.sin_port)));
        auto frame=client.grab();check(frame.valid());check(client.width==32 && client.height==24);
        check(client.pose.translation.x==1 && client.pose.translation.y==2 && client.pose.translation.z==3);
        auto rotation=client.pose.orientation.make<Eigen::Quaterniond>();
        check(rotation.angularDistance(Eigen::Quaterniond::Identity())<1e-9);
        client.command(true,rmcs::util::deg2rad(55),rmcs::util::deg2rad(-30),true);
        bool rejected=false;try { client.grab(); } catch(const std::exception&) { rejected=true; }
        check(rejected);
    }
    server.get();close(listener);std::cout<<"simulator protocol/geometry tests passed\n";
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
