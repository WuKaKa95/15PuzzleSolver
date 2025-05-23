#include <bits/stdc++.h>
using namespace std;
using Clock = chrono::steady_clock;

// Bail-out thresholds to prevent unbounded memory/time
static constexpr long long MAX_VISITS     = 15'000'000LL;
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
    vector<int> goalBoard(total);
    iota(goalBoard.begin(), goalBoard.end(), 1);
    goalBoard.back() = 0;

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

    // Best‐μ tracking
    long long mu = LLONG_MAX;
    shared_ptr<Node> bestF = nullptr, bestB = nullptr;

    while (!frontierForward.empty() && !frontierBackward.empty()) {
        bool expandForward = frontierForward.size() <= frontierBackward.size();

        auto &frontier     = expandForward ? frontierForward: frontierBackward;
        auto &gThisSide    = expandForward ? gScoreF : gScoreB;
        auto &gOtherSide   = expandForward ? gScoreB : gScoreF;
        auto &nodeMapThis  = expandForward ? nodeMapF : nodeMapB;
        auto &nodeMapOther = expandForward ? nodeMapB : nodeMapF;
        auto &closedThis   = expandForward ? closedForward: closedBackward;
        auto &heuristic    = expandForward ? heuristicForward : heuristicBackward;

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

        if (++visited > MAX_VISITS) {
            cerr << "[SOLVER] Aborting: too many visits\n";
            cout << "no_solution\n";
            return false;
        }
        if (frontierForward.size() + frontierBackward.size() > MAX_OPEN_SIZE) {
            cerr << "[SOLVER] Open‐queue too large – ";
            if (mu < LLONG_MAX && bestF && bestB) {
                cerr << "finishing with best meeting point\n";
                break;
            } else {
                cerr << "aborting cleanly\n";
                cout  << "no_solution\n";
                return false;
            }
        }

        // Record any meeting state and update μ
        if (gOtherSide.count(key)) {
            long long candidate = currentNode->gCost + gOtherSide[key];
            if (candidate < mu) {
                mu = candidate;
                if (expandForward) {
                    bestF = currentNode;
                    bestB = nodeMapOther[key];
                } else {
                    bestF = nodeMapOther[key];
                    bestB = currentNode;
                }
                // Prune forward frontier
                if (expandForward) {
                    vector<shared_ptr<Node>> survivors;
                    while (!frontierForward.empty()) {
                        auto n = frontierForward.top(); frontierForward.pop();
                        if (n->fCost < mu) survivors.push_back(n);
                    }
                    for (auto &n : survivors) frontierForward.push(n);
                }
                // Prune backward frontier
                else {
                    vector<shared_ptr<Node>> survivors;
                    while (!frontierBackward.empty()) {
                        auto n = frontierBackward.top(); frontierBackward.pop();
                        if (n->fCost < mu) survivors.push_back(n);
                    }
                    for (auto &n : survivors) frontierBackward.push(n);
                }
            }
        }
        // Check A* stopping criterion
        long long fFstar = frontierForward.empty()  ? LLONG_MAX : frontierForward.top()->fCost;
        long long fBstar = frontierBackward.empty() ? LLONG_MAX : frontierBackward.top()->fCost;
        if (min(fFstar, fBstar) >= mu) {
            break;
        }

        // Expand neighbors
        int blankRow    = currentNode->blankPos / boardSize;
        int blankColumn = currentNode->blankPos % boardSize;

        for (int dir = 0; dir < 4; ++dir) {
            int newRow = blankRow + dRow[dir];
            int newCol = blankColumn + dCol[dir];
            if (newRow < 0 || newRow >= boardSize
             || newCol < 0 || newCol >= boardSize) {
                continue;
             }

            int neighborBlank = newRow * boardSize + newCol;
            auto neighborBoard = currentNode->board;
            swap(neighborBoard[currentNode->blankPos],
                 neighborBoard[neighborBlank]);

            string neighborKey = boardToKey(neighborBoard);
            int tentativeG = currentNode->gCost + 1;

            // only consider if this gives a better g-score
            if (!gThisSide.count(neighborKey)
             || tentativeG < gThisSide[neighborKey])
            {
                int fCost = tentativeG + heuristic(neighborBoard);

                // Never enqueue any node whose fCost cannot beat mu
                if (fCost >= mu) {
                    continue;
                }

                gThisSide[neighborKey] = tentativeG;
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

        if (visited % 100000 == 0) {
            cerr << "[SOLVER] visited=" << visited
                 << " generated=" << generated
                 << " frontierF=" << frontierForward.size()
                 << " frontierB=" << frontierBackward.size()
                 << "\n";
        }
    }

    // Reconstruct using bestF & bestB
    if (mu < LLONG_MAX && bestF && bestB) {
        // Reconstruct forward half
        vector<vector<int>> pathF;
        for (auto p = bestF; p; p = p->parent)
            pathF.push_back(p->board);
        reverse(pathF.begin(), pathF.end());

        // Reconstruct backward half
        vector<vector<int>> pathB;
        for (auto p = bestB; p; p = p->parent)
            pathB.push_back(p->board);

        // Stitch & remove duplicate meeting state
        vector<vector<int>> fullPath = pathF;
        fullPath.insert(fullPath.end(), pathB.begin() + 1, pathB.end());

        // Output solution steps
        auto endTime = Clock::now();
        double elapsed = chrono::duration<double>(endTime - startTime).count();
        cout << "solution:\n";
        for (size_t i = 0; i < fullPath.size(); ++i) {
            cout << "Step " << i << ":\n";
            printBoard(fullPath[i], boardSize, cout);
            cout << "\n";
        }
        cout << "visited:   " << visited << "\n"
             << "generated: " << generated << "\n"
             << "path_length: " << fullPath.size() - 1 << "\n"
             << "time_s:    " << fixed << setprecision(3) << elapsed << "\n";
        cerr << "[SOLVER] Done (optimal bidir‐A*).\n";
        return true;
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
        for (int i = 0; i < totalTiles; ++i)
            startBoard[i] = stoi(argv[3 + i]);
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
        heuristicForward  = [&](const vector<int>& b) { return forwardHelper.hamming(b); };
        heuristicBackward = [&](const vector<int>& b) { return backwardHelper.hamming(b); };
        cerr << "Hamming heuristic.\n";
    } else if (heuristicName == "manhattan") {
        heuristicForward  = [&](const vector<int>& b) { return forwardHelper.combined(b); };
        heuristicBackward = [&](const vector<int>& b) { return backwardHelper.combined(b); };
        cerr << "Manhattan + Linear Conflict (combined) heuristic.\n";
    } else {
        cerr << "[SOLVER] Unknown heuristic: " << heuristicName << "\n";
        return 1;
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
