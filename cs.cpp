 /Basic Struct Definitions
#include <iostream>
#include <string>
using namespace std;

struct Book {
    string title;
    string author;
    string ISBN;
    double price;
    int pages;
};

int main() {
    Book b1 = {"C++ Primer", "Lippman", "12345", 49.99, 1024};
    Book b2 = {"Clean Code", "Martin", "67890", 39.99, 464};

    cout << "Book 1: " << b1.title << ", $" << b1.price << endl;
    cout << "Book 2: " << b2.title << ", $" << b2.price << endl;

    double avg = (b1.price + b2.price) / 2;
    cout << "Average price: $" << avg << endl;
    return 0;
}

//A – Parallel Arrays
string names[5];
int ids[5];
double scores[5][3];

//B – Array of Structs
struct Student {
    string name;
    int id;
    double scores[3];
};
Student students[5];

//C – Average Function:
//For struct version
double average(const Student &s) {
    return (s.scores[0] + s.scores[1] + s.scores[2]) / 3;
}

//3.Nested Structs
struct Address {
    string region, city, subCity, woreda, houseNumber;
};

struct Product {
    int productID;
    string name;
    double price;
    int stock;
};

struct Customer {
    string name, email, phoneNumber;
    Address address;
};

struct Order {
    int orderID;
    Customer customer;
    Product products[10];
    string orderDate;
    double totalAmount;
};

void createOrder(Order &o) { /* fill order */ }
double calculateTotal(Order &o) { /* sum product prices */ }
void displayOrder(const Order &o) { /* print details */ }

//4: Array of structs
struct Employee {
    int id;
    string name, department;
    double salary;
    int yearsOfService;
};

Employee employees[10];

// Functions:
Employee highestSalary(Employee arr[], int n);
double avgSalaryByDept(Employee arr[], int n, string dept);
void filterByService(Employee arr[], int n, int minYears);
void sortBySalary(Employee arr[], int n);
Employee* searchByID(Employee arr[], int n, int id);

//5: Passing structs to Functions
struct Rectangle {
    double length, width;
};

double calculateArea(Rectangle r) { // pass by value
    return r.length * r.width;
}

double calculatePerimeter(const Rectangle &r) { // const reference
    return 2 * (r.length + r.width);
}

void scale(Rectangle &r, double factor) { // reference to modify
    r.length *= factor;
    r.width *= factor;
}

void displayInfo(const Rectangle &r) { // const reference for read-only
    cout << "L: " << r.length << ", W: " << r.width << endl;
}

//6: First Class
class Book {
private:
    string title, author, ISBN;
    double price;
    int pages;
public:
    Book(string t, string a, string i, double p, int pg) : 
        title(t), author(a), ISBN(i), price(p), pages(pg) {}

    void displayInfo() const {
        cout << title << " by " << author << ", $" << price << endl;
    }
    void applyDiscount(double percent) {
        price *= (1 - percent/100);
    }
    // Getters/setters omitted for brevity
};

//7: Class with Member functions
class Student {
private:
    string name, studentID;
    double grades[5];
public:
    Student(string n, string id) : name(n), studentID(id) {
        for (int i = 0; i < 5; i++) grades[i] = 0;
    }
    bool addGrade(double grade, int index) {
        if (grade < 0 || grade > 100) return false;
        grades[index] = grade;
        return true;
    }
    double calculateGPA() const {
        double sum = 0;
        for (double g : grades) sum += g;
        return sum / 5;
    }
    // displayTranscript() etc.
};


//8: BankAccount
class BankAccount {
private:
    string accountNumber, accountHolder;
    double balance;
public:
    BankAccount(string acc, string holder, double bal) : 
        accountNumber(acc), accountHolder(holder), balance(bal) {}

    void deposit(double amount) { if (amount > 0) balance += amount; }
    bool withdraw(double amount) {
        if (amount > 0 && amount <= balance) {
            balance -= amount;
            return true;
        }
        return false;
    }
    double getBalance() const { return balance; }
    void display() const {
        cout << accountHolder << ": $" << balance << endl;
    }
};
// Random challenge: Direct access to private members causes compilation error.

//9: Library System
class Book {
private:
    string isbn, author;
    bool isCheckedOut;
public:
    Book(string i, string a) : isbn(i), author(a), isCheckedOut(false) {}
    void checkout() { isCheckedOut = true; }
    void returnBook() { isCheckedOut = false; }
    void display() const {
        cout << isbn << " by " << author << " - " 
             << (isCheckedOut ? "Checked out" : "Available") << endl;
    }
    string getISBN() const { return isbn; }
    bool getStatus() const { return isCheckedOut; }
};

class Library {
private:
    Book books[100];
    int bookCount;
public:
    Library() : bookCount(0) {}
    void addBook(Book b) { books[bookCount++] = b; }
    Book* findBook(string isbn) {
        for (int i = 0; i < bookCount; i++)
            if (books[i].getISBN() == isbn) return &books[i];
        return nullptr;
    }
    // Other functions implemented similarly
};

// 10: Struct to Class Evolution
class Employee {
private:
    int id, yearsOfService;
    string name, department;
    double salary;
public:
    Employee(int i, string n, string d, double s, int y) : 
        id(i), name(n), department(d), salary(s), yearsOfService(y) {}

    void giveRaise(double percent) { salary *= (1 + percent/100); }
    void promote(string newDept) { department = newDept; }
    double calculateBonus() const { 
        return (yearsOfService > 5) ? salary * 0.10 : 0; 
    }
    bool isEligibleForRetirement() const { return yearsOfService >= 20; }
};

//

class EmployeeDatabase {
private:
    Employee employees[100];
    int count;
public:
    void addEmployee(const Employee &e) { employees[count++] = e; }
    Employee* searchByID(int id) { /* linear search */ }
    void sortBySalary() { /* sorting logic */ }
    // Other functions from Problem 4
};
