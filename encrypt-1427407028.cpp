#include <stdio.h>
#include <fstream>
#include <string>
using namespace std;
void printFile(string &filename);
string entrypt(string &content){
	for(int i=0;i<content.size();i++){
		if('a'<=content[i]<'z'or'A'<=content[i]<'Z'){
		content[i]+=content[i];
	}
		else if(content[i]=='z'){
			content[i]='a';
		}else if(content[i]=='Z'){
			content[i]=='A';
		}
	}
	return content;
}

int main(int argc,char *argc[]){
	if(argc){
		string content;
		while(1){
			gets(content);
		}
		printFile(entrypt(content))
	}
	else{
		for(int i=1;i<argc;i++){
			print(argc[i]);		
		}
	}
}
void printFile(string &filename){
	ifstream myfile(filename);
	while(getline(infile,buff)){
		cout<<filename;
		cout<<entrypt(buff)<<endl;
	}
}

