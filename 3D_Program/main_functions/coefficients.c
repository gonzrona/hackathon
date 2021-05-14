
#include "../headers/structs.h"
#include "../headers/prototypes.h"

void derivative_kz_4th_order( int    Nz,
                              double hz,
                              double complex *k,
                              double complex *kz,
                              double          A,
                              double          B,
                              double          C,
                              double          z0,
                              double          z1 );
void derivative_kzz_4th_order( int    Nz,
                               double hz,
                               double complex *k,
                               double complex *kzz,
                               double          A,
                               double          B,
                               double          C,
                               double          z0,
                               double          z1 );

void second_order( System sys ) {
    int    i, Nz = sys.lat.Nz;
    double hx = sys.lat.hx, hy = sys.lat.hy, hz = sys.lat.hz;
    double complex *a = sys.a, *b = sys.b, *c = sys.c, *d = sys.d;
    double complex *am = sys.am, *bm = sys.bm, *cm = sys.cm, *dm = sys.dm;
    double complex *ap = sys.ap, *bp = sys.bp, *cp = sys.cp, *dp = sys.dp;
    double complex *k2_bg_ext = sys.k2_bg_ext;
    double          Rzx, Rzy;
    Rzx = hz * hz / hx / hx;
    Rzy = hz * hz / hy / hy;

    for ( i = 0; i < Nz; i++ ) {
        a[i] = 0;
        b[i] = Rzx;
        c[i] = Rzy;
        d[i] = -2.0 * ( Rzx + Rzy + 1.0 ) + hz * hz * k2_bg_ext[i + 1];

        ap[i] = 0;
        bp[i] = 0;
        cp[i] = 0;
        dp[i] = 1;

        am[i] = 0;
        bm[i] = 0;
        cm[i] = 0;
        dm[i] = 1;
    }
}

void fourth_order( System sys ) {
    int    i, Nz = sys.lat.Nz;
    double hx = sys.lat.hx, hy = sys.lat.hy, hz = sys.lat.hz;
    double complex *a = sys.a, *b = sys.b, *c = sys.c, *d = sys.d;
    double complex *am = sys.am, *bm = sys.bm, *cm = sys.cm, *dm = sys.dm;
    double complex *ap = sys.ap, *bp = sys.bp, *cp = sys.cp, *dp = sys.dp;
    double complex *k2_bg_ext = sys.k2_bg_ext;

    double Rzx, Rzy;
    double one_sixth   = 1.0 / 6.0;
    double one_twelfth = 1.0 / 12.0;
    Rzx                = hz * hz / hx / hx;
    Rzy                = hz * hz / hy / hy;

    for ( i = 0; i < Nz; i++ ) {
        a[i] = one_twelfth * ( Rzx + Rzy );
        b[i] = one_sixth * ( 4.0 * Rzx - Rzy - 1.0 + 0.5 * hz * hz * k2_bg_ext[i + 1] );
        c[i] = one_sixth * ( 4.0 * Rzy - Rzx - 1.0 + 0.5 * hz * hz * k2_bg_ext[i + 1] );
        d[i] = -4.0 * ( 1.0 + Rzx + Rzy ) / 3.0 + 0.5 * hz * hz * k2_bg_ext[i + 1];

        ap[i] = 0.0;
        bp[i] = one_twelfth * ( 1.0 + Rzx );
        cp[i] = one_twelfth * ( 1.0 + Rzy );
        dp[i] = 1.0 - one_sixth * ( 2.0 + Rzx + Rzy ) + one_twelfth * hz * hz * k2_bg_ext[i + 2];

        am[i] = 0.0;
        bm[i] = one_twelfth * ( 1.0 + Rzx );
        cm[i] = one_twelfth * ( 1.0 + Rzy );
        dm[i] = 1.0 - one_sixth * ( 2.0 + Rzx + Rzy ) + one_twelfth * hz * hz * k2_bg_ext[i];
    }
}

void sixth_order( System sys ) {

    int    i, Nz = sys.lat.Nz;
    double A = sys.A, B = sys.B, C = sys.C;
    double hz = sys.lat.hz, z0 = sys.lat.z0, z1 = sys.lat.z1;
    double complex *a = sys.a, *b = sys.b, *c = sys.c, *d = sys.d;
    double complex *am = sys.am, *bm = sys.bm, *cm = sys.cm, *dm = sys.dm;
    double complex *ap = sys.ap, *bp = sys.bp, *cp = sys.cp, *dp = sys.dp;
    double complex *k2_bg_ext = sys.k2_bg_ext;

    double complex *k_prime        = malloc( Nz * sizeof( double complex ) );
    double complex *k_double_prime = malloc( Nz * sizeof( double complex ) );

    derivative_kz_4th_order( Nz, hz, k2_bg_ext, k_prime, A, B, C, z0, z1 );
    derivative_kzz_4th_order( Nz, hz, k2_bg_ext, k_double_prime, A, B, C, z0, z1 );

    for ( i = 0; i < Nz; i++ ) {
        a[i] = 1.0 / 10.0 + hz * hz * k2_bg_ext[i + 1] / 90.0;
        b[i] = 7.0 / 15.0 - hz * hz * k2_bg_ext[i + 1] / 90.0;
        c[i] = 7.0 / 15.0 - hz * hz * k2_bg_ext[i + 1] / 90.0;
        d[i] = -64.0 / 15.0 + 14.0 * hz * hz * k2_bg_ext[i + 1] / 15.0 -
               hz * hz * hz * hz * k2_bg_ext[i + 1] * k2_bg_ext[i + 1] / 20.0 +
               hz * hz * hz * hz * k_double_prime[i] / 20.0;

        ap[i] = 1.0 / 30.0;
        bp[i] = 1.0 / 10.0 + hz * hz * k2_bg_ext[i + 2] / 90.0 + ( k_prime[i] * hz * hz * hz / 120.0 );
        cp[i] = 1.0 / 10.0 + hz * hz * k2_bg_ext[i + 2] / 90.0 + ( k_prime[i] * hz * hz * hz / 120.0 );
        dp[i] = 7.0 / 15.0 - hz * hz * k2_bg_ext[i + 2] / 90.0 +
                ( k_prime[i] * hz * hz * hz / 20.0 ) * ( 1.0 / 3.0 + k2_bg_ext[i + 2] * hz * hz / 6.0 );

        am[i] = 1.0 / 30.0;
        bm[i] = 1.0 / 10.0 + hz * hz * k2_bg_ext[i] / 90.0 - ( k_prime[i] * hz * hz * hz / 120.0 );
        cm[i] = 1.0 / 10.0 + hz * hz * k2_bg_ext[i] / 90.0 - ( k_prime[i] * hz * hz * hz / 120.0 );
        dm[i] = 7.0 / 15.0 - hz * hz * k2_bg_ext[i] / 90.0 -
                ( k_prime[i] * hz * hz * hz / 20.0 ) * ( 1.0 / 3.0 + k2_bg_ext[i] * hz * hz / 6.0 );
    }

    free( k_prime );
    k_prime = NULL;
    free( k_double_prime );
    k_double_prime = NULL;
}

void coefficients( System sys ) {
    if ( sys.order == sixth ) {
        sixth_order( sys );
    } else if ( sys.order == fourth ) {
        fourth_order( sys );
    } else {
        second_order( sys );
    }
}
