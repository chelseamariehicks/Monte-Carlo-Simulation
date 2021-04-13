/**********************************************************************************
 * Name: Chelsea Marie Hicks
 * ONID 931286984
 * Course: CS 475
 * Assignment: Project 1
 * Due Date: April 16, 2021 by 11:59 PM
 * 
 * Description: Monte Carlo simulation to determine probability of a cannon
 * hitting the castle and destroying it.
 *
 * Resources include: CS475 documentation -- program essentially provided
 * Compiled on macOS using g++-10 -o monte monte.cpp -O3 -lm -fopenmp
 * Compiled on flip using g++ -o monte monte.cpp -O3 -lm -fopenmp
***********************************************************************************/

#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

//Print debugging messages
#ifndef DEBUG
#define DEBUG       false
#endif

/*
//Set number of threads
#ifndef NUMT
#define NUMT        1
#endif

//Set number of trials in monte carlo simulation
#ifndef NUMTRIALS
#define NUMTRIALS       50000
#endif
*/

//Set number of tries to find maximum performance
#ifndef NUMTRIES
#define NUMTRIES        10
#endif

int NUMT = 1;
int NUMTRIALS = 1;

//Ranges for the random numbers:
const float GMIN =	20.0;	// ground distance in meters
const float GMAX =	30.0;	// ground distance in meters
const float HMIN =	10.0;	// cliff height in meters
const float HMAX =	20.0;	// cliff height in meters
const float DMIN  =	10.0;	// distance to castle in meters
const float DMAX  =	20.0;	// distance to castle in meters
const float VMIN  =	10.0;	// intial cnnonball velocity in meters / sec
const float VMAX  =	30.0;	// intial cnnonball velocity in meters / sec
const float THMIN = 30.0;	// cannonball launch angle in degrees
const float THMAX =	70.0;	// cannonball launch angle in degrees

const float GRAVITY =	-9.8;	// acceleraion due to gravity in meters / sec^2
const float TOL = 5.0;		// tolerance in cannonball hitting the castle in meters
				// castle is destroyed if cannonball lands between d-TOL and d+TOL

//Helper Functions
float Ranf(float low, float high) {
    float r = (float) rand();
    float t = r / (float) RAND_MAX;

    return low + t * (high - low);
}

int Ranf(int ilow, int ihigh) {
    float low = (float) ilow;
    float high = ceil((float) ihigh);

    return (int) Ranf(low, high);
}

void TimeOfDaySeed() {
    struct tm y2k = { 0 };
    y2k.tm_hour = 0;
    y2k.tm_min = 0;
    y2k.tm_sec = 0;
    y2k.tm_year = 100;
    y2k.tm_mon = 0;
    y2k.tm_mday = 1;

    time_t timer;
    time(&timer);
    double seconds = difftime(timer, mktime(&y2k));
    unsigned int seed = (unsigned int) (1000.*seconds); //milli
    srand(seed);
}

//Function prototypes:
float       Ranf(float, float);
int         Ranf(int, int);
void        TimeOfDaySeed();

//Functions to convert degrees to radians
inline float Radians(float d) {
    return (M_PI/180.f) * d;
}

//Main program
int main(int argc, char *argv[]) {

#ifndef _OPENMP
    fprintf(stderr, "No OpenMP support!\n");
    return 1;
#endif

    if(argc >= 2) {
        NUMT = atoi(argv[1]);
    }

    if(argc >= 3) {
        NUMTRIALS = atoi(argv[2]);
    }

    //Seed the random number generator
    TimeOfDaySeed();

    //Set the number of threads
    omp_set_num_threads(NUMT);

    float* vs = new float[NUMTRIALS];
    float* ths = new float[NUMTRIALS];
    float* gs = new float[NUMTRIALS];
    float* hs = new float[NUMTRIALS];
    float* ds = new float[NUMTRIALS];

    //Fill in the random value arrays
    for(int n = 0; n < NUMTRIALS; n++) {
        vs[n] = Ranf(VMIN, VMAX);
        ths[n] = Ranf(THMIN, THMAX);
        gs[n] = Ranf(GMIN, GMAX);
        hs[n] = Ranf(HMIN, HMAX);
        ds[n] = Ranf(DMIN, DMAX);
    }

    //Prepare to record the max performance and probability
    double maxPerformance = 0.;
    int numHits;

    //Looking for max performance
    for(int tries = 0; tries < NUMTRIES; tries++) {
        double time0 = omp_get_wtime();

        numHits = 0;

        #pragma omp parallel for shared(NUMTRIALS) reduction(+:numHits)
        for(int n = 0; n < NUMTRIALS; n++) {
            //randomize everything
            float v = vs[n];
            float thr = Radians(ths[n]);
            float vx = v * cos(thr);
            float vy = v * sin(thr);
            float g = gs[n];
            float h = hs[n];
            float d = ds[n];

            //See if the ball doesn't even reach the cliff
            float t = (-2. * vy) / GRAVITY;
            float x = vx * t;

            if(x <= g) {
                if(DEBUG) {
                    fprintf(stderr, "Ball doesn't even reach the cliff\n");
                }
            }
            else {
                //See if the ball hits the cliff face
                float t = g / vx;
                float y = vy * t + (0.5 * GRAVITY * pow(t, 2.));
                if(y <= h) {
                    if(DEBUG) {
                        fprintf(stderr, "Ball hits the cliff face\n");
                    }
                }
                else {
                    //Ball hits the upper deck
                    //the time solution for this is quadratic equation of the form:
                    //at^2 + bt + c = 0
                    //where 'a' multiplies time^2
                    //      'b' multiples time
                    //      'c' is a constant
                    float a = 0.5 * GRAVITY;
                    float b = vy;
                    float c = -h;
                    float disc = b * b - 4.f * a * c; //quadratic formula discriminant

                    //Ball doesn't go as high as the upper deck:
                    //this should "never happen"...
                    if(disc < 0.) {
                        if(DEBUG) {
                            fprintf(stderr, "Ball doesn't reach upper deck.\n");
                            exit(1); //something is wrong...
                        }
                    }

                    //Ball successfully hits the ground above the cliff:
                    //get the intersection:
                    disc = sqrtf(disc);
                    float t1 = (-b + disc) / (2.f * a);   //time to intersect high ground
                    float t2 = (-b - disc) / (2.f * a);   //time to intersect high ground

                    //only care about the second intersection
                    float tmax = t1;
                    if(t2 > t1) {
                        tmax = t2;
                    }

                    //How far does the ball land horizontally from the edge of the cliff?
                    float upperDist = vx * tmax - g;

                    //See if the ball hits the castle
                    if(fabs(upperDist - d) > TOL) {
                        if(DEBUG) {
                            fprintf(stderr, "Misses the castle at upperDist = %8.3f\n", upperDist);
                        }
                    }
                    else {
                        if(DEBUG) {
                            fprintf(stderr, "Hits the castle at upperDist = %8.3f\n", upperDist);
                        }
                        numHits++;
                    }  
                }//if ball clears cliff face
            }//if ball gets as far as cliff face
        }  //for (# monte carlo trials)
        double time1 = omp_get_wtime();
        double megaTrialsPerSecond = (double) NUMTRIALS / (time1 - time0) / 1000000.;
        if(megaTrialsPerSecond > maxPerformance) {
            maxPerformance = megaTrialsPerSecond;
        }
    }//for (# of timing trials)

    float probability = (float) numHits / (float) (NUMTRIALS); //just get for last NUMTRIES run
    fprintf(stderr, "%2d threads: %8d trials; probability = %6.2f%%; megatrials/sec = %6.21f\n", 
        NUMT, NUMTRIALS, 100.*probability, maxPerformance);

    return 0;
}