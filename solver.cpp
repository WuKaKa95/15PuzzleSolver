// solver.cpp
#include <bits/stdc++.h>
using namespace std;
using Clock = chrono::steady_clock;

// Bail-out thresholds to prevent unbounded memory/time
static constexpr long long MAX_VISITS     = 50'000'000LL;
static constexpr size_t    MAX_OPEN_SIZE = 10'000'000;

// Called when operator new fails
void outOfMemoryHandler() {
    cerr << "[SOLVER] Out of memory – exiting cleanly.\n";
    cout << "no_solution\n";
    exit(0);
}

// Serialize a board state to a string key for hashing
string boardToKey(const vector<int>& board) {
    ostringstream oss;
    for (int tile : board) {
        oss << tile << ',';
    }
    return oss.str();
}

// Print a board in a human-readable grid
void printBoard(const vector<int>& board, int size, ostream& os) {
    for (int row = 0; row < size; ++row) {
        for (int col = 0; col < size; ++col) {
            os << setw(2)
               << board[row * size + col]
               << (col + 1 == size ? '\n' : ' ');
        }
    }
}

struct HeuristicHelper {
    vector<int> targetPositions;
    int size;

    HeuristicHelper(const vector<int>& targetBoard)
        : size(sqrt(targetBoard.size()))
    {
        int total = size * size;
        targetPositions.resize(total + 1); // +1 to handle tile values up to total
        for (int i = 0; i < total; ++i) {
            int tile = targetBoard[i];
            targetPositions[tile] = i;
        }
    }

    int hamming(const vector<int>& board) const {
        int count = 0;
        for (int i = 0; i < (int)board.size(); ++i) {
            int tile = board[i];
            if (tile != 0 && targetPositions[tile] != i) {
                ++count;
            }
        }
        return count;
    }

    int manhattan(const vector<int>& board) const {
        int total = 0;
        for (int i = 0; i < (int)board.size(); ++i) {
            int tile = board[i];
            if (tile == 0) continue;
            int goalIndex = targetPositions[tile];
            total += abs(i / size - goalIndex / size)
                   + abs(i % size - goalIndex % size);
        }
        return total;
    }

    int linearConflict(const vector<int>& board) const {
        int conflicts = 0;

        // Row conflicts
        for (int r = 0; r < size; ++r) {
            vector<int> rowTiles;
            for (int c = 0; c < size; ++c) {
                int pos = r * size + c;
                int tile = board[pos];
                if (tile == 0) continue;
                int targetRow = targetPositions[tile] / size;
                if (targetRow == r) {
                    rowTiles.push_back(targetPositions[tile]);
                }
            }
            for (int i = 0; i < (int)rowTiles.size(); ++i) {
                for (int j = i + 1; j < (int)rowTiles.size(); ++j) {
                    if (rowTiles[i] % size > rowTiles[j] % size) {
                        ++conflicts;
                    }
                }
            }
        }

        // Column conflicts
        for (int c = 0; c < size; ++c) {
            vector<int> colTiles;
            for (int r = 0; r < size; ++r) {
                int pos = r * size + c;
                int tile = board[pos];
                if (tile == 0) continue;
                int targetCol = targetPositions[tile] % size;
                if (targetCol == c) {
                    colTiles.push_back(targetPositions[tile]);
                }
            }
            for (int i = 0; i < (int)colTiles.size(); ++i) {
                for (int j = i + 1; j < (int)colTiles.size(); ++j) {
                    if (colTiles[i] / size > colTiles[j] / size) {
                        ++conflicts;
                    }
                }
            }
        }

        return 2 * conflicts;
    }

    int combined(const vector<int>& board) const {
        return manhattan(board) + linearConflict(board);
    }
};

// Check if a board configuration is solvable
bool isSolvable(const vector<int>& board) {
    int size = sqrt(board.size());
    int inversions = 0;

    for (int i = 0; i < (int)board.size(); ++i) {
        if (board[i] == 0) continue;
        for (int j = i + 1; j < (int)board.size(); ++j) {
            if (board[j] != 0 && board[i] > board[j]) {
                ++inversions;
            }
        }
    }

    if (size % 2 == 1) {
        return inversions % 2 == 0;
    } else {
        int rowFromBottom = size
            - ((int)(find(board.begin(), board.end(), 0) - board.begin()) / size);
        return (inversions + rowFromBottom) % 2 == 1;
    }
}

// Read board from file: first line is N, then N×N tiles
vector<int> readBoardFromFile(const string& filename, int& size) {
    ifstream in(filename);
    in >> size;
    vector<int> board(size * size);
    for (int i = 0; i < size * size; ++i) {
        in >> board[i];
    }
    return board;
}

// A* search node
struct Node {
    vector<int> board;
    int blankPos;
    int gCost;
    int fCost;
    shared_ptr<Node> parent;
};

// Bidirectional A* search
bool bidirectionalAStar(
    const vector<int>& startBoard,
    int boardSize,
    const function<int(const vector<int>&)>& heuristicForward,
    const function<int(const vector<int>&)>& heuristicBackward)
{
    int total = boardSize * boardSize;
    // Goal state: [1,2,...,N*N-1,0]
    vector<int> goalBoard(total);
    iota(goalBoard.begin(), goalBoard.end(), 1);
    goalBoard.back() = 0;

    // Min-heap ordered by lowest fCost
    auto cmp = [](auto &a, auto &b){ return a->fCost > b->fCost; };
    priority_queue<shared_ptr<Node>, vector<shared_ptr<Node>>, decltype(cmp)>
        frontierForward(cmp), frontierBackward(cmp);

    unordered_map<string,int> gScoreF, gScoreB;
    unordered_map<string,shared_ptr<Node>> nodeMapF, nodeMapB;
    unordered_set<string> closedForward, closedBackward;

    // Initialize forward search
    int startBlank = find(startBoard.begin(), startBoard.end(), 0)
                   - startBoard.begin();
    auto rootF = make_shared<Node>(
        Node{startBoard, startBlank, 0,
             heuristicForward(startBoard), nullptr});
    frontierForward.push(rootF);
    gScoreF[boardToKey(startBoard)] = 0;
    nodeMapF[boardToKey(startBoard)] = rootF;

    // Initialize backward search
    int goalBlank = total - 1;
    auto rootB = make_shared<Node>(
        Node{goalBoard, goalBlank, 0,
             heuristicBackward(goalBoard), nullptr});
    frontierBackward.push(rootB);
    gScoreB[boardToKey(goalBoard)] = 0;
    nodeMapB[boardToKey(goalBoard)] = rootB;

    vector<int> dRow = {-1,1,0,0}, dCol = {0,0,-1,1};
    long long visited = 0, generated = 0;
    auto startTime = Clock::now();

    while (!frontierForward.empty() && !frontierBackward.empty()) {
        bool expandForward = frontierForward.size() <= frontierBackward.size();

        auto &frontier     = expandForward ? frontierForward
                                          : frontierBackward;
        auto &gThisSide    = expandForward ? gScoreF : gScoreB;
        auto &gOtherSide   = expandForward ? gScoreB : gScoreF;
        auto &nodeMapThis  = expandForward ? nodeMapF : nodeMapB;
        auto &nodeMapOther = expandForward ? nodeMapB : nodeMapF;
        auto &closedThis   = expandForward ? closedForward
                                          : closedBackward;
        auto &heurestic    = expandForward ? heuristicForward : heuristicBackward;

        auto currentNode = frontier.top();
        frontier.pop();
        string key = boardToKey(currentNode->board);

        // Skip stale or already closed
        if (currentNode->gCost > gThisSide[key]
         || closedThis.count(key))
        {
            continue;
        }
        closedThis.insert(key);

        // Bail-out conditions
        if (++visited > MAX_VISITS) {
            cerr << "[SOLVER] Aborting: too many visits\n";
            cout << "no_solution\n";
            return false;
        }
        if (frontierForward.size() + frontierBackward.size()
          > MAX_OPEN_SIZE)
        {
            cerr << "[SOLVER] Aborting: open-queue too large\n";
            cout << "no_solution\n";
            return false;
        }

        // Meet‐in‐the‐middle check
        if (gOtherSide.count(key)) {
            // Reconstruct path from start to meeting point
            vector<vector<int>> pathFromStart;
            for (auto p = currentNode; p; p = p->parent) {
                pathFromStart.push_back(p->board);
            }
            reverse(pathFromStart.begin(), pathFromStart.end());

            // Reconstruct path from goal to meeting point
            auto meetNode = nodeMapOther[key];
            vector<vector<int>> pathFromGoal;
            for (auto p = meetNode; p; p = p->parent) {
                pathFromGoal.push_back(p->board);
            }

            // Correct backward path direction
            if (!expandForward) {
                reverse(pathFromGoal.begin(), pathFromGoal.end()); // Now pathFromGoal is [start -> ... -> meet]
                reverse(pathFromStart.begin(), pathFromStart.end()); // Now pathFromStart is [meet -> ... -> goal]
            }

            // Combine paths and remove duplicate meet node
            vector<vector<int>> fullPath;
            if (expandForward) {
                fullPath = pathFromStart;
                fullPath.insert(fullPath.end(), pathFromGoal.begin() + 1, pathFromGoal.end());
            } else {
                fullPath = pathFromGoal; // start -> meet
                fullPath.insert(fullPath.end(), pathFromStart.begin() + 1, pathFromStart.end()); // meet -> goal
            }

            // Output solution steps
            auto endTime = Clock::now();
            double elapsed = chrono::duration<double>(
                                 endTime - startTime).count();

            cout << "solution:\n";
            for (size_t step = 0; step < fullPath.size(); ++step) {
                cout << "Step " << step << ":\n";
                printBoard(fullPath[step], boardSize, cout);
                cout << "\n";
            }
            cout << "visited:   " << visited << "\n"
                 << "generated: " << generated << "\n"
                 << "path_length: " << fullPath.size() - 1 << "\n"
                 << "time_s:    " << fixed << setprecision(3)
                                  << elapsed << "\n";
            cerr << "[SOLVER] Done (bidirectional).\n";
            return true;
        }

        // Expand neighbors
        int blankRow = currentNode->blankPos / boardSize;
        int blankColumn = currentNode->blankPos % boardSize;
        for (int dir = 0; dir < 4; ++dir) {
            int blankNewRow = blankRow + dRow[dir], blankNewColumn = blankColumn + dCol[dir];
            if (blankNewRow < 0 || blankNewRow >= boardSize
             || blankNewColumn < 0 || blankNewColumn >= boardSize)
            {
                continue;
            }
            int neighborBlank = blankNewRow * boardSize + blankNewColumn;
            auto neighborBoard = currentNode->board;
            swap(neighborBoard[currentNode->blankPos],
                 neighborBoard[neighborBlank]);

            string neighborKey = boardToKey(neighborBoard);
            int tentativeG = currentNode->gCost + 1;

            if (!gThisSide.count(neighborKey)
             || tentativeG < gThisSide[neighborKey])
            {
                gThisSide[neighborKey] = tentativeG;
                int fCost = tentativeG + heurestic(neighborBoard);
                auto neighborNode = make_shared<Node>(
                    Node{neighborBoard,
                         neighborBlank,
                         tentativeG,
                         fCost,
                         currentNode});
                nodeMapThis[neighborKey] = neighborNode;
                frontier.push(neighborNode);
                ++generated;
            }
        }

        // Periodic progress log
        if (visited % 100000 == 0) {
            cerr << "[SOLVER] visited=" << visited
                 << " generated=" << generated
                 << " frontierF=" << frontierForward.size()
                 << " frontierB=" << frontierBackward.size()
                 << "\n";
        }
    }

    // No solution found
    cout << "no_solution\n";
    cerr << "[SOLVER] Exhausted both frontiers.\n";
    return false;
}

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    set_new_handler(outOfMemoryHandler);

    bool fromFile = false;
    string inputFile, heuristicName;
    int boardSize, totalTiles;
    vector<int> startBoard;

    if (argc >= 4 && string(argv[1]) == "--input-file") {
        fromFile = true;
        inputFile = argv[2];
        heuristicName = argv[3];
        startBoard = readBoardFromFile(inputFile, boardSize);
    }
    else {
        if (argc < 4) {
            cerr << "[SOLVER] Usage:\n"
                 << "  " << argv[0]
                 << " [--input-file file] <hamming|manhattan> <N> <tiles...>\n";
            return 1;
        }
        heuristicName = argv[1];
        boardSize     = stoi(argv[2]);
        totalTiles    = boardSize * boardSize;
        startBoard.resize(totalTiles);
        for (int i = 0; i < totalTiles; ++i) {
            startBoard[i] = stoi(argv[3 + i]);
        }
    }

    totalTiles = boardSize * boardSize;
    cerr << "[SOLVER] Bidirectional A* with ";

    // Prepare goal board
    vector<int> goalBoard(totalTiles);
    iota(goalBoard.begin(), goalBoard.end(), 1);
    goalBoard.back() = 0;

    // Create heuristic helpers
    HeuristicHelper forwardHelper(goalBoard);
    HeuristicHelper backwardHelper(startBoard);

    function<int(const vector<int>&)> heuristicForward, heuristicBackward;

    if (heuristicName == "hamming") {
        heuristicForward = [forwardHelper](const vector<int>& b) { return forwardHelper.hamming(b); };
        heuristicBackward = [backwardHelper](const vector<int>& b) { return backwardHelper.hamming(b); };
        cerr << "Hamming\n";
    } else {
        heuristicForward = [forwardHelper](const vector<int>& b) { return forwardHelper.combined(b); };
        heuristicBackward = [backwardHelper](const vector<int>& b) { return backwardHelper.combined(b); };
        cerr << "Manhattan + LinearConflict\n";
    }

    cerr << "[SOLVER] Start board:\n";
    printBoard(startBoard, boardSize, cerr);

    if ((int)startBoard.size() != totalTiles) {
        cerr << "[SOLVER] Invalid input size\n";
        return 1;
    }
    if (!isSolvable(startBoard)) {
        cout << "unsolvable\n";
        cerr << "[SOLVER] Puzzle not solvable\n";
        return 0;
    }

    bool solved = bidirectionalAStar(
        startBoard, boardSize, heuristicForward, heuristicBackward);
    return solved ? 0 : 1;
}