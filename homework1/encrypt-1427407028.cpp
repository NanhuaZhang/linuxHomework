#include <stdio.h>
#include <fstream>
#include <string>
#include <iostream>
using namespace std;
string entrypt(string &content) {
	for (int i = 0; i<content.size(); i++) {
		if (content[i] == 'z'||content[i] == 'Z') {
			content[i] -= 25;
		}		
		else if (('a' <= content[i]&& content[i]<'z')||('A' <= content[i]&& content[i]<'Z')){
			content[i] += 1;
		}
	}
	return content;
}
void printFile(string filename) {
	ifstream myfile(filename);
	string buff;
	while (getline(myfile, buff)) {
		cout << entrypt(buff) << endl;
	}
	myfile.close();
}
int main(int argc, char const *argv[]) {
		if (argc==1) {
			string content="";
			while (cin>>content,!cin.eof()) {
				cout<<entrypt(content)<<endl;
			}
		}
		else {
			for (int i = 1; i<argc; i++) {
				printFile(argv[i]);
			}
		}
}

