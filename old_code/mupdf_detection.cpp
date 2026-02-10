#include <bits/stdc++.h>

using namespace std;

int main(int argc, char** argv){
    ifstream file(argv[1]);
    std::set<string> errors;
    
    string line;
    string pattern = "MuPDF";
    while(getline(file, line)) {
        if (line.find(pattern) != std::string::npos) errors.insert(line);
    }
    for (auto it : errors) cout << it << endl;
    file.close();
}