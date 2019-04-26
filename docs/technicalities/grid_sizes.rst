Optimal transverse grid sizes
=============================

FFT works best for grid sizes that are factorizable into small numbers.
Any size will work, but the performance may vary dramatically.

FFTW documentation quotes the optimal size for their algorithm as
:math:`2^a 3^b 5^c 7^d 11^e 13^f`,
where :math:`e+f` is either :math:`0` or :math:`1`,
and the other exponents are arbitrary.

While LCODE 3D does not use FFTW (it uses ``cufft`` instead, wrapped by ``cupy``),
the formula is still quite a good rule of thumb
for calculating performance-friendly ``config_example.grid_steps`` values.

The employed FFT sizes for a grid sized :math:`N` are :math:`2N-2`
for both DST (:func:`dst2d`, :func:`mix2d`) and DCT transforms (:func:`dct2d`, :func:`mix2d`)
when we take padding and perimeter cell stripping into account.

This leaves us to find such :math:`N` that :math:`N-1` satisfies the small-factor conditions.

If you don't mind arbitrary grid sizes, we suggest using

1. :math:`N=2^K + 1`, they always perform the best, or

2. one of the roundish
   :math:`201`, :math:`301`, :math:`401`, :math:`501`, :math:`601`, :math:`701`, :math:`801`, :math:`901`,
   :math:`1001`, :math:`1101`, :math:`1201`, :math:`1301`, :math:`1401`, :math:`1501`, :math:`1601`, :math:`1801`,
   :math:`2001`, :math:`2101`, :math:`2201`, :math:`2401`, :math:`2501`, :math:`2601`, :math:`2701`, :math:`2801`,
   :math:`3001`, :math:`3201`, :math:`3301`, :math:`3501`, :math:`3601`, :math:`3901`, :math:`4001`.

The code to check for the FFTW criteria above and some of the matching numbers are listed below.

.. code-block:: python

   def factorize(n, a=[]):
       if n <= 1:
           return a
       for i in range(2, n + 1):
           if n % i == 0:
               return factorize(n // i, a + [i])

   def good_size(n):
       factors = factorize(n - 1)
       return (all([f in [2, 3, 4, 5, 7, 11, 13] for f in factors])
               and actors.count(11) + factors.count(13) < 2 and
               and n % 2)

   ', '.join([str(a) for a in range(20, 4100) if good_size(a)])

:math:`21`, :math:`23`, :math:`25`, :math:`27`, :math:`29`, :math:`31`, :math:`33`, :math:`37`, :math:`41`, :math:`43`, :math:`45`, :math:`49`, :math:`51`, :math:`53`, :math:`55`, :math:`57`, :math:`61`, :math:`65`, :math:`67`, :math:`71`, :math:`73`, :math:`79`, :math:`81`, :math:`85`, :math:`89`, :math:`91`, :math:`97`, :math:`99`, :math:`101`, :math:`105`, :math:`109`, :math:`111`, :math:`113`, :math:`121`, :math:`127`, :math:`129`, :math:`131`, :math:`133`, :math:`141`, :math:`145`, :math:`151`, :math:`155`, :math:`157`, :math:`161`, :math:`163`, :math:`169`, :math:`177`, :math:`181`, :math:`183`, :math:`193`, :math:`197`, :math:`199`, :math:`201`, :math:`209`, :math:`211`, :math:`217`, :math:`221`, :math:`225`, :math:`235`, :math:`241`, :math:`251`, :math:`253`, :math:`257`, :math:`261`, :math:`265`, :math:`271`, :math:`281`, :math:`289`, :math:`295`, :math:`301`, :math:`309`, :math:`313`, :math:`321`, :math:`325`, :math:`331`, :math:`337`, :math:`351`, :math:`353`, :math:`361`, :math:`365`, :math:`379`, :math:`385`, :math:`391`, :math:`393`, :math:`397`, :math:`401`, :math:`417`, :math:`421`, :math:`433`, :math:`441`, :math:`449`, :math:`451`, :math:`463`, :math:`469`, :math:`481`, :math:`487`, :math:`491`, :math:`501`, :math:`505`, :math:`513`, :math:`521`, :math:`529`, :math:`541`, :math:`547`, :math:`551`, :math:`561`, :math:`577`, :math:`589`, :math:`595`, :math:`601`, :math:`617`, :math:`625`, :math:`631`, :math:`641`, :math:`649`, :math:`651`, :math:`661`, :math:`673`, :math:`687`, :math:`701`, :math:`703`, :math:`705`, :math:`721`, :math:`729`, :math:`751`, :math:`757`, :math:`769`, :math:`771`, :math:`781`, :math:`785`, :math:`793`, :math:`801`, :math:`811`, :math:`833`, :math:`841`, :math:`865`, :math:`881`, :math:`883`, :math:`897`, :math:`901`, :math:`911`, :math:`925`, :math:`937`, :math:`961`, :math:`973`, :math:`981`, :math:`991`, :math:`1001`, :math:`1009`, :math:`1025`, :math:`1041`, :math:`1051`, :math:`1057`, :math:`1079`, :math:`1081`, :math:`1093`, :math:`1101`, :math:`1121`, :math:`1135`, :math:`1153`, :math:`1171`, :math:`1177`, :math:`1189`, :math:`1201`, :math:`1233`, :math:`1249`, :math:`1251`, :math:`1261`, :math:`1275`, :math:`1281`, :math:`1297`, :math:`1301`, :math:`1321`, :math:`1345`, :math:`1351`, :math:`1373`, :math:`1387`, :math:`1401`, :math:`1405`, :math:`1409`, :math:`1441`, :math:`1457`, :math:`1459`, :math:`1471`, :math:`1501`, :math:`1513`, :math:`1537`, :math:`1541`, :math:`1561`, :math:`1569`, :math:`1585`, :math:`1601`, :math:`1621`, :math:`1639`, :math:`1651`, :math:`1665`, :math:`1681`, :math:`1729`, :math:`1751`, :math:`1761`, :math:`1765`, :math:`1783`, :math:`1793`, :math:`1801`, :math:`1821`, :math:`1849`, :math:`1873`, :math:`1891`, :math:`1921`, :math:`1945`, :math:`1951`, :math:`1961`, :math:`1981`, :math:`2001`, :math:`2017`, :math:`2049`, :math:`2059`, :math:`2081`, :math:`2101`, :math:`2107`, :math:`2113`, :math:`2157`, :math:`2161`, :math:`2185`, :math:`2201`, :math:`2241`, :math:`2251`, :math:`2269`, :math:`2305`, :math:`2311`, :math:`2341`, :math:`2353`, :math:`2377`, :math:`2401`, :math:`2431`, :math:`2451`, :math:`2465`, :math:`2497`, :math:`2501`, :math:`2521`, :math:`2549`, :math:`2561`, :math:`2593`, :math:`2601`, :math:`2641`, :math:`2647`, :math:`2689`, :math:`2701`, :math:`2731`, :math:`2745`, :math:`2751`, :math:`2773`, :math:`2801`, :math:`2809`, :math:`2817`, :math:`2881`, :math:`2913`, :math:`2917`, :math:`2941`, :math:`2971`, :math:`3001`, :math:`3025`, :math:`3073`, :math:`3081`, :math:`3121`, :math:`3137`, :math:`3151`, :math:`3169`, :math:`3201`, :math:`3235`, :math:`3241`, :math:`3251`, :math:`3277`, :math:`3301`, :math:`3329`, :math:`3361`, :math:`3403`, :math:`3431`, :math:`3457`, :math:`3501`, :math:`3511`, :math:`3521`, :math:`3529`, :math:`3565`, :math:`3585`, :math:`3601`, :math:`3641`, :math:`3697`, :math:`3745`, :math:`3751`, :math:`3781`, :math:`3823`, :math:`3841`, :math:`3851`, :math:`3889`, :math:`3901`, :math:`3921`, :math:`3961`, :math:`4001`, :math:`4033`, :math:`4051`, :math:`4097`
