// g++ -std=c++11 main_sincruce.cpp -o tsp_sin -O2
// ./tsp_sin Practica2/datos/chr12a.dat 20 10000 normal

#include <cstdlib>
#include <iostream>
#include <map>
#include <vector>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <queue>
#include <string>
#include <sstream>

using namespace std;

typedef vector<vector<int> > Matriz; 
int fileSize = 0; // tamaño del cromosoma
int tam = 0; // tamaño de la poblacion
////////////////////////////////////////////////////////////////////////////
void to_pobl_string(vector<vector<int> > poblacion){
	
	for(int i = 0 ; i < tam ; i++){
	   for(int j = 0 ; j < fileSize ; j ++){
		cout << poblacion[i][j] << " ";	 
	  }

		cout << endl;
	}

}
////////////////////////////////////////////////////////////////////////////
void to_vector_string(vector<int>  individuo){
	
	for(int j = 0 ; j < individuo.size() ; j ++){
		cout << individuo[j] << " ";	 
	}

	cout << endl;
	

}

//////////////////////////////////////////////////////
class CLParser{
	public:
		CLParser(int argc_, char * argv_[],bool switches_on_=false);
		~CLParser(){}
		string get_arg(int i);
		string get_arg(string s);
	private:
		int argc;
		vector<string> argv;
		bool switches_on;
		map<string,string> switch_map;
};

CLParser::CLParser(int argc_, char * argv_[],bool switches_on_){
	argc = argc_;
	argv.resize(argc);
	copy(argv_, argv_ + argc, argv.begin());
	switches_on = switches_on_;
	if(switches_on){
		vector<string>::iterator it1, it2;
		it1 = argv.begin();
		it2 = it1 + 1;
		while(true){
			if(it1 == argv.end())
				break;
			if(it2 == argv.end())
				break;
			if((*it1)[0] == '-')
				switch_map[*it1]=*(it2);
			it1++;
			it2++;
		}
	}
}

string CLParser::get_arg(int i){
	if(i >= 0 && i < argc)
		return argv[i];
	return "";
}

string CLParser::get_arg(string s){
	if(!switches_on)
		return "";
	if(switch_map.find(s) != switch_map.end())
		return switch_map[s];
	return "";
}

///////////////////////////////////////////////////////////////////////////////


void leer_puntos(string& nombre, vector<vector <int>> & flujo, vector<vector <int>> & distancia){
	ifstream datos;
	datos.open(nombre.c_str());
	if(datos.is_open()){
		vector <int> aux;
		datos >> fileSize;
		char conv[128];
	        for(int i = 0; i < fileSize; ++i){
	            for(int j = 0; j < fileSize; ++j){
			datos >> conv;
 		        aux.push_back(atoi(conv));
	            }
		    flujo.push_back(aux);
		    aux.clear();
	        }
	        for(int i = 0; i < fileSize; ++i){
	            for(int j=0; j < fileSize ; ++j){
			datos >> conv;
 		        aux.push_back(atoi(conv));
	            }
		    distancia.push_back(aux);
		    aux.clear();
	        }
		
		datos.close();
	}else{ 
		cout << "Error de Lectura de puntos en " << nombre << endl;
	}
 }

///////////////////////////////////////////////////////////////////////////7

void pob_inicial(string& tampob, vector<vector<int> > & poblacion){
	tam = stoi(tampob);
	vector <int> aux;
        for(int i = 0; i < tam; ++i){
            for(int j = 0; j < fileSize; ++j){
		aux.push_back(j);
            }
	    std::random_shuffle(aux.begin(), aux.end());
	    poblacion.push_back(aux);
	    aux.clear();
        }
}

////////////////////////////////////////////////////////////////////////////
int fitness(vector<int> cromosoma,vector<vector<int> > & distancia, vector<vector<int> > & flujo){
	int f;
	f = 0;
	int tamanio;
	tamanio = cromosoma.size();
	for( int i = 0; i < tamanio ; i++) {		
 		for( int j = 0; j < tamanio ; j++) {
 			f = f + (distancia[i][j] * flujo[cromosoma[i]][cromosoma[j]]);
 		}
	}
	return -f;
}

////////////////////////////////////////////////////////////////////////////
vector<int> allFitness(vector<vector<int> > &poblacion,vector<vector<int> > & distancia, vector<vector<int> > & flujo){
	vector<int> fitnes;
	fitnes.reserve(tam);
	for( int i = 0; i < tam ; i++) {		
 		fitnes.push_back(fitness(poblacion[i], distancia, flujo));
	}
	return fitnes;
}

////////////////////////////////////////////////////////////////////////////
// recogo la posicion de los seleccionados. seleccion por torneo.
vector<int> seleccion(float probselect,vector<int> &fitness){
	int probseleccion = 0;
	probseleccion = (int) (probselect * tam);
	vector<int> selecc;
	selecc.reserve(probseleccion);
	for( int i = 0; i < probseleccion ; i++) {		
 		int select1 = 0;
		int select2 = 0;
		int mejorselect = 0;
		select1 = rand() % tam;
		do {
			select2 = rand() % tam;
		}while(select1==select2);

		if(fitness[select1]>fitness[select2]){
			mejorselect = select1;
		}else{
			mejorselect = select2;
		}
		// controla que no haya individuos repetidos, si el seleccionado ya está entre los elegidos damos otra ronda.
		if(std::find(selecc.begin(), selecc.end(), mejorselect) != selecc.end()){
			i--;
		}else{
			selecc.push_back(mejorselect);
		}
	}
	return selecc;
}
////////////////////////////////////////////////////////////////////////////
vector<int> cruce(vector<int> padre1, vector<int> padre2){
	int crucep1;
	crucep1 = padre1.size()/3;
	vector<int> p1;
	p1.reserve(crucep1);
	vector<int> hijo;
	hijo.reserve(padre1.size());
	// introducimos la tercera parte del primer padre
	for (int i = 0 ; i < crucep1 ; i++) {
		hijo.push_back(padre1[i]);
		p1.push_back(padre1[i]);
	}
	// introducimos el resto en el orden del padre 2 
	for (int i = 0 ; i < padre2.size() ; i++) {
		if(std::find(p1.begin(), p1.end(), padre2[i]) == p1.end()) {
			hijo.push_back(padre2[i]);
		}
	}
	p1.clear();
	return hijo;
}
////////////////////////////////////////////////////////////////////////////
void mutacion(vector<int> & mutado){
	int azar1, azar2;
	azar1 = rand() % mutado.size();
	azar2 = rand() % mutado.size();
	if (azar1 == azar2) {
		if (azar2 > (mutado.size() - 1)) {
		azar2 = azar2 - 1;
		}
		if (azar2 < 1) {
		azar2 = azar2 + 1;
		}
	}
	swap(mutado[azar1],mutado[azar2]);
}
////////////////////////////////////////////////////////////////////////////
void reemplazo(vector<int> & mejor_individuo,  vector<int> & fitness, vector<vector<int> > & popu, vector<int> & seleccionados){
	vector<vector<int> > 	nuevapoblacion;
	// conservamos la mejor solución encontrada hasta el momento
	nuevapoblacion.push_back(mejor_individuo);
	//nueva población con el mejor y el peor de todos
	int mejor,peor;
	mejor = 0;
	peor = 0;
	int fitnessmejor, fitnesspeor = fitness[0];
	for(int j = 0 ; j < tam ; j++) {
		if(fitness[j]> fitnessmejor){
			mejor = j;
			fitnessmejor = fitness[j];
		}
		if(fitness[j]< fitnesspeor){
			peor = j;
			fitnesspeor = fitness[j];
		}
	}
	nuevapoblacion.push_back(popu[mejor]);
	nuevapoblacion.push_back(popu[peor]);

	//introduzco los hijos seleccionados 
	for(int j = 0 ; j < seleccionados.size() ; j++) {
		if(nuevapoblacion.size()<popu.size()){
			nuevapoblacion.push_back(popu[seleccionados[j]]);
		}
	}
	//relleno de los padres hasta completar tamaño
	int a = 0;
	while(nuevapoblacion.size() < popu.size()){
		nuevapoblacion.push_back(popu[a]);
		a++;
	}
	
	//reemplazo poblacion
	popu = nuevapoblacion;
	nuevapoblacion.clear();
}
////////////////////////////////////////////////////////////////////////////
int mejorfitness(vector<int>& mejor_individuo, vector<int> & fitness,vector<vector<int> > popu){
	int fitnessmejor = fitness[0];
	for(int j = 0 ; j < tam ; j++) {
		if(fitness[j]> fitnessmejor){
			fitnessmejor = fitness[j];
			mejor_individuo = popu[j];
		}
	}
	return -fitnessmejor;
}
////////////////////////////////////////////////////////////////////////////
vector <int> greedyOptimoLocal(vector <int > & individuo){
    	vector <int > salida;
    	vector <int > candidato;
    	int aleatorio1, pos_ale1, aleatorio2, pos_ale2;
    	int gen1, gen2;

    	for (int i=0; i<individuo.size(); i++){
    	  salida.push_back(individuo[i]);
    	  candidato.push_back(i);
    	}

    	aleatorio1 = rand() % (candidato.size()-1) ;
    	pos_ale1 = candidato[aleatorio1];

    	candidato.erase(candidato.begin()+aleatorio1);
    	aleatorio2 = rand() % (candidato.size()-1) ;
    	pos_ale2 = candidato[aleatorio2];

    	gen1 = individuo[pos_ale1];
    	gen2 = individuo[pos_ale2];

    	salida[pos_ale2] = gen1;
    	salida[pos_ale1] = gen2;
    	return salida;
}
////////////////////////////////////////////////////////////////////////////
vector<int> allFitnessBaldwi(vector<vector<int> > &poblacion,vector<vector<int> > & distancia, vector<vector<int> > & flujo){
	vector<int> fitnes;
	fitnes.reserve(tam);
	vector<int> pupu;
	int f1=0,f2=0;
	int iteracion=0;

	for( int i = 0; i < tam ; i++) {		
 		pupu=greedyOptimoLocal(poblacion[i]);
		f1=fitness(pupu,distancia,flujo);
		f2 = fitness(poblacion[i],distancia,flujo);
		if(f1 > f2){
			fitnes.push_back(f1);
		}else{
			fitnes.push_back(f2);
		}
	}
	pupu.clear();
	return fitnes;
}
////////////////////////////////////////////////////////////////////////////
void learnLamarck(vector<vector<int> > &poblacion,vector<vector<int> > & distancia, vector<vector<int> > & flujo, int pivote){
	vector<int> pupu;
	int iteracion=0;
	do{
		for( int i = 0; i < tam ; i++) {		
	 		pupu=greedyOptimoLocal(poblacion[i]);
			if(fitness(pupu,distancia,flujo) > fitness(poblacion[i],distancia,flujo)){
				poblacion[i] = pupu;
			}
		}
		iteracion ++ ;
	}while(iteracion<pivote);
	pupu.clear();
}
////////////////////////////////////////////////////////////////////////////
int main( int argc, char* argv[] ){
// srand (time(NULL)); // quitar semilla, para que sea todo aleatorio
	Matriz flujo, distancia, poblacion;
	string fp, tampob, opcion;
	int iteraciones = 0 ;
	float probselect = 0.8;
	if(argc < 5){
		cout << "Error Formato: ./a.out tai256c.dat 10_individuos 100_iteraciones normal/lamarck/baldwi" << endl;
		exit(1);
	}
	//lee el fichero y rellena vector de flujo y distancias
	CLParser cmd_line(argc, argv, false);
	fp = cmd_line.get_arg(1);
	leer_puntos(fp,flujo, distancia);
	tampob = cmd_line.get_arg(2);
	iteraciones = atoi(argv[3]);
	opcion = cmd_line.get_arg(4);

	pob_inicial(tampob,poblacion);
	to_pobl_string(poblacion);
	double time; // para medir el tiempo
 	unsigned start, end;
	double totaltime = 0.0;

	if (opcion == "normal"){

		vector<int> mejor_individuo = poblacion[0]; 
		vector<int> allFit = allFitness(poblacion, distancia, flujo);
		cout << mejorfitness(mejor_individuo,allFit,poblacion) << "" ;

	   for(int p = 0 ; p < iteraciones ; p++){
		start = clock();

		// selecciono los mejores (vector de posiciones)
		vector<int> seleccionados = seleccion(probselect,allFit);

		// mutan los hijos
		for(int i = 0 ; i < seleccionados.size(); i++){
			mutacion(poblacion[seleccionados[i]]);	
		}
		// inserto descendencia a población
		reemplazo(mejor_individuo,allFit,poblacion,seleccionados);

		// evaluo cada individuo
		vector<int> allFit = allFitness(poblacion, distancia, flujo);
		// guardo el mejor fitness y el mejor individuo
		if (p%10 == 0){
			cout  << ","<< mejorfitness(mejor_individuo,allFit,poblacion) ;
		}else{
			mejorfitness(mejor_individuo,allFit,poblacion);
		}

		allFit.clear();
		seleccionados.clear();

		end = clock();
        	time = (double(start-end)/CLOCKS_PER_SEC);
		//cout << time << " tiempo iteracion" << endl;
		totaltime = totaltime + time;
	   }
	   cout << " tiempo total "<< totaltime  << endl;
	   cout << " mejor individuo " << ""; to_vector_string(mejor_individuo);
	   cout << " fitness individuo " <<  fitness(mejor_individuo, distancia, flujo)<< endl;

	}else if(opcion == "lamarck"){
		// evaluo cada individuo
		vector<int> mejor_individuo = poblacion[0]; 
		vector<int> allFit = allFitness(poblacion, distancia, flujo);
		cout << mejorfitness(mejor_individuo,allFit,poblacion) << "" ;

	   for(int p = 0 ; p < iteraciones ; p++){
		start = clock();

		// la población aprende y cambia
		learnLamarck(poblacion, distancia, flujo,6);

		// evaluo cada individuo
		allFit = allFitness(poblacion, distancia, flujo);

		// guardo el mejor individuo y el mejor fitness que muestro en pantalla
		if (p%10 == 0){
			cout  << ","<< mejorfitness(mejor_individuo,allFit,poblacion) ;
		}else{
			mejorfitness(mejor_individuo,allFit,poblacion);
		}

		// selecciono los mejores (vector de posiciones)
		vector<int> seleccionados = seleccion(probselect,allFit);
		// mutan los hijos
		for(int i = 0 ; i < seleccionados.size(); i++){
			mutacion(poblacion[seleccionados[i]]);	
		}
		// inserto descendencia a población
		reemplazo(mejor_individuo,allFit,poblacion,seleccionados);
		

		allFit.clear();
		seleccionados.clear();

		end = clock();
        	time = (double(start-end)/CLOCKS_PER_SEC);
		//cout << time << " tiempo iteracion" << endl;
		totaltime = totaltime + time;
	   }
	   cout << " tiempo total "<< totaltime  << endl;
	   cout << " mejor individuo " << ""; to_vector_string(mejor_individuo);
	   cout << " fitness individuo " <<  fitness(mejor_individuo, distancia, flujo)<< endl;

	}else if(opcion == "baldwi"){

		vector<int> mejor_individuo = poblacion[0]; 
		vector<int> allFit= allFitness(poblacion, distancia, flujo);
		cout << mejorfitness(mejor_individuo,allFit,poblacion) << "" ;

	   for(int p = 0 ; p < iteraciones ; p++){
		start = clock();

		// evaluo cada individuo
		allFit = allFitnessBaldwi(poblacion, distancia, flujo);
		// guardo el mejor fitness y el mejor individuo
		if (p%10 == 0){
			cout  << ","<< mejorfitness(mejor_individuo,allFit,poblacion) ;
		}else{
			mejorfitness(mejor_individuo,allFit,poblacion);
		}

		// selecciono los mejores (vector de posiciones)
		vector<int> seleccionados = seleccion(probselect,allFit);

		// mutan los hijos
		for(int i = 0 ; i < seleccionados.size(); i++){
			mutacion(poblacion[seleccionados[i]]);	
		}
		// inserto descendencia a población
		reemplazo(mejor_individuo,allFit,poblacion,seleccionados);

		allFit.clear();
		seleccionados.clear();

		end = clock();
        	time = (double(start-end)/CLOCKS_PER_SEC);
		//cout << time << " tiempo iteracion" << endl;
		totaltime = totaltime + time;
	   }
	   cout << " tiempo total "<< totaltime  << endl;
	   cout << " mejor individuo " << ""; to_vector_string(mejor_individuo);
	   cout << " fitness individuo " <<  fitness(mejor_individuo, distancia, flujo)<< endl;

	}
		
	
}
