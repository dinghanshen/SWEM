#include <iostream>
using namespace std;

int main ()
{
    char* pGetResultsManagerDir;
    pGetResultsManagerDir = getenv("GIT_RESULTS_MANAGER_DIR");

    if (pGetResultsManagerDir == NULL) {
        cout << "Environment variable GIT_RESULTS_MANAGER_DIR is undefined. To demonstrate logging, run this instead as" << endl;
        cout << "    gitresman junk ./demo-c" << endl;;
    } else {
        printf("The current GIT_RESULTS_MANAGER_DIR is: %s\n", pGetResultsManagerDir);
    }

    cout << "This line is logged" << endl;
    cerr << "This line is logged (stderr)" << endl;

    cout << "This line is logged" << endl;
    cerr << "This line is logged (stderr)" << endl;

    cout << "This line is logged" << endl;
    cerr << "This line is logged (stderr)" << endl;

    return 0;
}
