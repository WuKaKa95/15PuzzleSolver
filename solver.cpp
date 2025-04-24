// solver.cpp
#include <bits/stdc++.h>
using namespace std;
using Clock = chrono::steady_clock;

// bail‐out thresholds
static constexpr long long MAX_VISITS     = 50'000'000LL;
static constexpr size_t    MAX_OPEN_SIZE = 10'000'000;

// OOM handler
void oom_handler(){
    cerr<<"[SOLVER] Out of memory – exiting cleanly.\n";
    cout<<"no_solution\n";
    exit(0);
}

// board → string key
string board_key(const vector<int>& B){
    ostringstream oss;
    for(int x: B) oss<<x<<',';
    return oss.str();
}

// pretty‐print a board
void print_board(const vector<int>& B,int N,ostream& os){
    for(int r=0;r<N;r++){
        for(int c=0;c<N;c++){
            os<<setw(2)<<B[r*N+c]<<(c+1==N?'\n':' ');
        }
    }
}

// Hamming
int hamming(const vector<int>& B){
    int cnt=0, N=sqrt(B.size());
    for(int i=0;i<(int)B.size();i++)
        if(B[i] && B[i]!=(i+1)) cnt++;
    return cnt;
}

// Manhattan
int manhattan(const vector<int>& B){
    int dist=0, N=sqrt(B.size());
    for(int i=0;i<(int)B.size();i++) if(B[i]){
        int v=B[i]-1;
        dist += abs(i/N - v/N) + abs(i%N - v%N);
    }
    return dist;
}

// Linear conflicts
int linear_conflict(const vector<int>& B){
    int N=sqrt(B.size()), lc=0;
    // rows
    for(int r=0;r<N;r++){
        vector<int> row;
        for(int c=0;c<N;c++){
            int v=B[r*N+c];
            if(v && (v-1)/N==r) row.push_back(v-1);
        }
        for(int i=0;i<(int)row.size();i++)
            for(int j=i+1;j<(int)row.size();j++)
                if(row[i]%N>row[j]%N) lc++;
    }
    // cols
    for(int c=0;c<N;c++){
        vector<int> col;
        for(int r=0;r<N;r++){
            int v=B[r*N+c];
            if(v && (v-1)%N==c) col.push_back(v-1);
        }
        for(int i=0;i<(int)col.size();i++)
            for(int j=i+1;j<(int)col.size();j++)
                if(col[i]/N>col[j]/N) lc++;
    }
    return 2*lc;
}

// Manhattan + LC
int manhattan_lc(const vector<int>& B){
    return manhattan(B) + linear_conflict(B);
}

// solvability
bool is_solvable(const vector<int>& B){
    int N=sqrt(B.size()), inv=0;
    for(int i=0;i<(int)B.size();i++) if(B[i])
        for(int j=i+1;j<(int)B.size();j++)
            if(B[j] && B[i]>B[j]) inv++;
    if(N%2==1) return inv%2==0;
    int row = N - (find(B.begin(),B.end(),0)-B.begin())/N;
    return (inv+row)%2==1;
}

// read from file
vector<int> read_board_file(const string& p,int& N){
    ifstream in(p);
    in>>N;
    vector<int> B(N*N);
    for(int i=0;i<N*N;i++) in>>B[i];
    return B;
}

// Node type
struct Node {
    vector<int> A;
    int blank, g, f;
    shared_ptr<Node> parent;
};

// Bidirectional A*1f
bool bidir_astar(const vector<int>& start,int N,
                 function<int(const vector<int>&)> hfunc){
    int total = N*N;
    vector<int> GOAL(total);
    iota(GOAL.begin(),GOAL.end(),1);
    GOAL.back() = 0;

    auto cmp = [](auto &a, auto &b){ return a->f > b->f; };
    priority_queue< shared_ptr<Node>, vector<shared_ptr<Node>>, decltype(cmp) >
        openF(cmp), openB(cmp);

    unordered_map<string, int> gF,gB;
    unordered_map<string,shared_ptr<Node>> nodeF,nodeB;
    unordered_set<string> closedF,closedB;

    int zb = find(start.begin(), start.end(), 0) - start.begin();
    auto rF = make_shared<Node>(Node{start, zb, 0, hfunc(start), nullptr});
    gF[board_key(start)] = 0; nodeF[board_key(start)] = rF; openF.push(rF);

    int zg = total-1;
    auto rB = make_shared<Node>(Node{GOAL, zg, 0, hfunc(GOAL), nullptr});
    gB[board_key(GOAL)] = 0; nodeB[board_key(GOAL)] = rB; openB.push(rB);

    vector<int> dr={-1,1,0,0}, dc={0,0,-1,1};
    long long visited=0, generated=0;
    auto t0 = Clock::now();

    while(!openF.empty() && !openB.empty()){
        bool forward = openF.size() <= openB.size();
        auto &openQ    = forward ? openF  : openB;
        auto &gThis    = forward ? gF     : gB;
        auto &gOther   = forward ? gB     : gF;
        auto &nodeThis = forward ? nodeF  : nodeB;
        auto &nodeOther= forward ? nodeB  : nodeF;
        auto &closedT  = forward ? closedF: closedB;

        auto cur = openQ.top(); openQ.pop();
        string key = board_key(cur->A);
        if(cur->g>gThis[key] || closedT.count(key)) continue;
        closedT.insert(key);

        if(++visited > MAX_VISITS){
            cerr<<"[SOLVER] Aborting: too many visits\n";
            cout<<"no_solution\n";
            return false;
        }
        if(openF.size()+openB.size()>MAX_OPEN_SIZE){
            cerr<<"[SOLVER] Aborting: open‐queue too large\n";
            cout<<"no_solution\n";
            return false;
        }

        if(gOther.count(key)){
            // build forward+backward paths
            vector<vector<int>> pathF, pathB;
            auto meetF = cur;
            auto meetB = nodeOther[key];
            for(auto p=meetF; p; p=p->parent) pathF.push_back(p->A);
            reverse(pathF.begin(), pathF.end());
            for(auto p=meetB; p; p=p->parent) pathB.push_back(p->A);

            // merge into fullPath
            vector<vector<int>> fullPath = pathF;
            for(size_t i=1; i<pathB.size(); ++i)
                fullPath.push_back(pathB[i]);

            auto t1 = Clock::now();
            double elapsed = chrono::duration<double>(t1-t0).count();

            // print step by step
            cout<<"solution:\n";
            for(size_t i=0; i<fullPath.size(); ++i){
                cout<<"Step "<<i<<":\n";
                print_board(fullPath[i], N, cout);
                cout<<"\n";
            }
            cout<<"visited: "<<visited<<"\n"
                <<"generated: "<<generated<<"\n"
                <<"path_length: "<<fullPath.size()-1<<"\n"
                <<"time_s: "<<fixed<<setprecision(3)<<elapsed<<"\n";
            cerr<<"[SOLVER] Done (bidirectional).\n";
            return true;
        }

        int r=cur->blank/N, c=cur->blank%N;
        for(int k=0;k<4;k++){
            int nr=r+dr[k], nc=c+dc[k];
            if(nr<0||nr>=N||nc<0||nc>=N) continue;
            int ni = nr*N + nc;
            auto B2 = cur->A; swap(B2[cur->blank], B2[ni]);
            string k2 = board_key(B2);
            int g2 = cur->g + 1;
            if(!gThis.count(k2) || g2 < gThis[k2]){
                gThis[k2] = g2;
                auto node = make_shared<Node>(Node{B2, ni, g2, g2 + hfunc(B2), cur});
                nodeThis[k2] = node;
                openQ.push(node);
                generated++;
            }
        }

        if(visited % 100000 == 0){
            cerr<<"[SOLVER] visited="<<visited
                <<" generated="<<generated
                <<" openF="<<openF.size()
                <<" openB="<<openB.size()<<"\n";
        }
    }

    cout<<"no_solution\n";
    cerr<<"[SOLVER] Exhausted both frontiers.\n";
    return false;
}

int main(int argc,char** argv){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    set_new_handler(oom_handler);

    bool from_file=false;
    string input_file, hname;
    int N, total;
    vector<int> start;

    if(argc>=4 && string(argv[1])=="--input-file"){
        from_file=true;
        input_file=argv[2];
        hname=argv[3];
        start=read_board_file(input_file,N);
    } else {
        if(argc<4){
            cerr<<"[SOLVER] Usage:\n"
                <<"  "<<argv[0]<<" [--input-file file] <hamming|manhattan> <N> <tiles...>\n";
            return 1;
        }
        hname=argv[1];
        N=stoi(argv[2]);
        total=N*N;
        start.resize(total);
        for(int i=0;i<total;i++) start[i]=stoi(argv[3+i]);
    }
    total=N*N;

    cerr<<"[SOLVER] Bidirectional A* with ";
    function<int(const vector<int>&)> hfunc;
    if(hname=="hamming"){
        hfunc = hamming;
        cerr<<"Hamming\n";
    } else {
        hfunc = manhattan_lc;
        cerr<<"Manhattan+LinearConflict\n";
    }

    cerr<<"[SOLVER] Start board:\n"; print_board(start,N,cerr);

    if((int)start.size()!=total){
        cerr<<"[SOLVER] Invalid input size\n"; return 1;
    }
    if(!is_solvable(start)){
        cout<<"unsolvable\n";
        cerr<<"[SOLVER] Not solvable\n";
        return 0;
    }

    bool ok = bidir_astar(start, N, hfunc);
    return ok ? 0 : 1;
}
