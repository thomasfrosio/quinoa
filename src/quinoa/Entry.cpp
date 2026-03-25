#include <noa/Runtime.hpp>
#include <noa/Session.hpp>
#include <noa/Signal.hpp>

#include "Plot.hpp"
#include "quinoa/align/Align.hpp"
#include "quinoa/ctf/CTF.hpp"

#include "quinoa/ExcludeViews.hpp"
#include "quinoa/Logger.hpp"
#include "quinoa/Metadata.hpp"
#include "quinoa/Settings.hpp"
#include "quinoa/Stack.hpp"
#include "quinoa/Thickness.hpp"
#include "quinoa/Reconstruct.hpp"

namespace {
    using namespace qn;

    // auto test01() {
    //     const auto spacing = 0.2;
    //     const auto angles = noa::deg2rad(Vec{0., 60., 60.});
    //     const auto plane_rotation = ( // TODO check
    //         nx::rotate_z(angles[0]) *
    //         nx::rotate_y(angles[1]) *
    //         nx::rotate_x(angles[2])
    //     );
    //     const auto plane_normal = (plane_rotation * Vec{1., 0., 0.}).as<f32>();
    //
    //     auto image = Array<f32>({1, 1, 100, 100});
    //     auto span = image.span_contiguous<f32, 2>();
    //     noa::iwise(span.shape(), image.device(), [&](Vec<isize, 2> indices) {
    //         auto coordinates = indices.as<f32>() - (image.shape().filter(2, 3).vec / 2).as<f32>();
    //         const auto& [c, b, a] = plane_normal;
    //         const auto volume_z_coordinate = -(a * coordinates[1] + b * coordinates[0]) / c;
    //         const auto volume_z_coordinate_nm = volume_z_coordinate * 1;
    //         span(indices) = volume_z_coordinate_nm;
    //     });
    //
    //     noa::write_image(image, "~/Tmp/image_z.mrc");
    // }

    // auto test02() {
    //     auto ctf = CTFAnisotropic64::Parameters{
    //         .pixel_size = {2., 2.},
    //         .defocus = {2.5, 0.2, 0.},
    //         .voltage = 300.,
    //         .amplitude = 0.1,
    //         .cs = 2.7,
    //         .phase_shift = 0,
    //         .bfactor = 0,
    //         .scale = 1,
    //     }.to_ctf();
    //     auto ctf_iso = CTFIsotropic64(ctf);
    //
    //     f64 wedge = 3.75;
    //     f64 half = wedge / 2;
    //     i32 bin_size = 23;
    //     auto linspace = noa::Linspace{.start = 45 - half, .stop = 45 + half, .endpoint = true}.for_size(bin_size);
    //
    //     auto shape = Shape4{1, 1, 1, 10000};
    //     auto sim = Array<f64>(shape.rfft());
    //     auto sum = like(sim);
    //     for (i32 i{0}; i < bin_size; ++i) {
    //         auto phi = noa::deg2rad(45.);
    //         ctf_iso.set_defocus(ctf.defocus_at(phi));
    //         ns::ctf_isotropic<"h">(sim, shape, ctf_iso, {.ctf_squared = true});
    //         noa::ewise(sim, sum, [](f64 i, f64& o) { o += i; });
    //     }
    //     noa::ewise({}, sum, [&](f64& o) { o /= static_cast<f64>(bin_size); });
    //     save_plot_xy(noa::Linspace{0., 0.5}, sum, "~/Tmp/quinoa/ctf_paper/bin.txt", {.label = "iso"});
    //
    //     fill(sum, 0);
    //     for (i32 i{0}; i < bin_size; ++i) {
    //         fmt::println("phi={}", linspace(i));
    //         auto phi = noa::deg2rad(linspace(i));
    //         ctf_iso.set_defocus(ctf.defocus_at(phi));
    //         ns::ctf_isotropic<"h">(sim, shape, ctf_iso, {.ctf_squared = true});
    //         noa::ewise(sim, sum, [](f64 i, f64& o) { o += i; });
    //     }
    //     noa::ewise({}, sum, [&](f64& o) { o /= static_cast<f64>(bin_size); });
    //     save_plot_xy(noa::Linspace{0., 0.5}, sum, "~/Tmp/quinoa/ctf_paper/bin.txt", {.label = "aniso"});
    //
    //     // f64 diff{};
    //     // for (i32 i{1}; i < bin_size; ++i) {
    //     //     auto d0 = ctf.defocus_at(linspace(i));
    //     //     auto d1 = ctf.defocus_at(linspace(i - 1));
    //     //     diff = std::max(std::abs(d0 - d1), diff);
    //     // }
    //     // fmt::println("diff={}", diff);
    // }
    //
    // void test03() {
    //     Vec<f64, 2> fftfreq_range{0.0,0.5};
    //     auto spectrum = Array<f32>::from_values(
    //         1.6109443, 1.5546821, 1.4547946, 1.385552, 1.2988318, 1.2396648, 1.1889594, 1.1317582,
    //         1.1008685, 1.075424, 1.0633056, 1.0786837, 1.0742517, 1.091464, 1.1006866, 1.1225797,
    //         1.1422756, 1.1598911, 1.1906793, 1.2178274, 1.2065622, 1.2139487, 1.2008574, 1.1867713,
    //         1.1736801, 1.1533579, 1.1254344, 1.0987773, 1.0970638, 1.0862366, 1.072636, 1.07852,
    //         1.0837504, 1.1002266, 1.1128113, 1.135092, 1.1418402, 1.1601408, 1.1675472, 1.166621,
    //         1.1499225, 1.1420625, 1.12643, 1.1070334, 1.0920968, 1.0874481, 1.0916533, 1.1044347,
    //         1.1230835, 1.1457226, 1.1593333, 1.1655923, 1.1678499, 1.1632249, 1.1501868, 1.1340433,
    //         1.1136937, 1.1024458, 1.0977496, 1.1013119, 1.1225953, 1.1423001, 1.1540701, 1.15829,
    //         1.1604947, 1.1648873, 1.1549032, 1.1411545, 1.1335194, 1.123729, 1.1182994, 1.1301097,
    //         1.1462486, 1.1664089, 1.1670399, 1.1670965, 1.1595436, 1.1425846, 1.1351662, 1.127137,
    //         1.1332753, 1.1514468, 1.1590216, 1.172121, 1.1658584, 1.1655383, 1.1566094, 1.1512712,
    //         1.1466782, 1.1510472, 1.1607213, 1.1710284, 1.1762879, 1.1798768, 1.1770579, 1.1641631,
    //         1.1608558, 1.1575329, 1.1601161, 1.1721137, 1.1751851, 1.1820447, 1.1862108, 1.173991,
    //         1.1655749, 1.1663257, 1.1702803, 1.1710658, 1.1876445, 1.1814268, 1.1860224, 1.1822551,
    //         1.1731497, 1.1749034, 1.1755728, 1.1877359, 1.1979003, 1.1862509, 1.1828568, 1.1754155,
    //         1.1731699, 1.175337, 1.1804593, 1.1924816, 1.1976736, 1.1833177, 1.1917126, 1.1813755,
    //         1.1850128, 1.1958666, 1.2028873, 1.2012835, 1.2015907, 1.1894641, 1.1887271, 1.1925639,
    //         1.1945444, 1.1967006, 1.1955448, 1.1945573, 1.1897439, 1.19049, 1.1925145, 1.2004788,
    //         1.2006342, 1.1947365, 1.1964993, 1.1971359, 1.1935917, 1.1975546, 1.2027136, 1.208487,
    //         1.2004935, 1.1864631, 1.1983228, 1.2066381, 1.2088418, 1.2043576, 1.204012, 1.1998186,
    //         1.2051404, 1.2117513, 1.2089972, 1.2070404, 1.2048312, 1.2021046, 1.2064478, 1.2095531,
    //         1.1999156, 1.2026551, 1.1998478, 1.2027373, 1.205614, 1.2121007, 1.2028525, 1.2048038,
    //         1.2083993, 1.2158148, 1.2101493, 1.2101951, 1.210645, 1.2056048, 1.2106595, 1.2162647,
    //         1.2114367, 1.2080646, 1.2090466, 1.2094215, 1.2111716, 1.2131577, 1.2149882, 1.2110875,
    //         1.209482, 1.2115812, 1.2203904, 1.21341, 1.2071483, 1.2103359, 1.2097507, 1.2133936,
    //         1.2051752, 1.2088381, 1.2132106, 1.2115428, 1.2141342, 1.2118831, 1.2095494, 1.2175906,
    //         1.2135034, 1.210228, 1.2148091, 1.2083151, 1.2144488, 1.2161915, 1.2132454, 1.206671,
    //         1.2102883, 1.2134302, 1.2118244, 1.2078633, 1.173157,
    //         1.1, 1.0, 0.9, 0.5, 0.3, 0.2, 0.1, 0., 0., 0., 0.);
    //
    //     auto spectrum2 = Array<f32>::from_values(
    //         1.9748614, 1.9058369, 1.8166975, 1.7412736, 1.6588042, 1.6171422, 1.5623451, 1.5214922, 1.5166644,
    //         1.5272813, 1.530596, 1.555525, 1.5864542, 1.6032449, 1.6442387, 1.6756061, 1.7163619, 1.7261353, 1.7194669,
    //         1.7202995, 1.7223856, 1.6880667, 1.6813262, 1.6496364, 1.6160139, 1.5980414, 1.5621653, 1.5242671,
    //         1.4966588, 1.5006527, 1.5170195, 1.5336132, 1.5469596, 1.5663203, 1.5750811, 1.6092811, 1.6209013,
    //         1.6238045, 1.6137273, 1.6011682, 1.5929927, 1.5721968, 1.5295506, 1.5224767, 1.502961, 1.5040253, 1.5001299,
    //         1.5161275, 1.5256052, 1.5600337, 1.5746821, 1.5578239, 1.5697179, 1.5535249, 1.5347682, 1.5286179,
    //         1.5112822, 1.4734373, 1.4738724, 1.4868681, 1.500263, 1.4915648, 1.4996557, 1.5123043, 1.509603, 1.5227866,
    //         1.5188694, 1.4909858, 1.4824501, 1.4698205, 1.4539921, 1.4605901, 1.4615636, 1.4631348, 1.477968, 1.4663305,
    //         1.4588562, 1.4681679, 1.4448307, 1.4407208, 1.4290397, 1.4298849, 1.4181566, 1.4043722, 1.422667, 1.4290929,
    //         1.4251866, 1.411474, 1.4024096, 1.3925188, 1.3825277, 1.3793101, 1.384603, 1.3742207, 1.372133, 1.367654,
    //         1.3611186, 1.3592029, 1.3584611, 1.3412899, 1.3434935, 1.3406858, 1.3396529, 1.3349391, 1.3349235,
    //         1.3296525, 1.3266697, 1.3091103, 1.3040116, 1.3022305, 1.2836758, 1.2937545, 1.3002336, 1.2899139,
    //         1.2697787, 1.26596, 1.25734, 1.2615404, 1.2548422, 1.2514993, 1.2468262, 1.2506105, 1.2396804, 1.2243434,
    //         1.225367, 1.2277269, 1.2134948, 1.2208159, 1.2138047, 1.2078389, 1.2035508, 1.2013066, 1.1807127, 1.1833982,
    //         1.1845986, 1.1814028, 1.1744589, 1.1650689, 1.1581484, 1.155502, 1.1517507, 1.1502624, 1.1496081, 1.1429789,
    //         1.1309925, 1.1320128, 1.1293914, 1.1315278, 1.1285824, 1.1227043, 1.1073344, 1.1063796, 1.1113533,
    //         1.1033764, 1.09393, 1.0880363, 1.0825039, 1.0811753, 1.0793598, 1.0730608, 1.0700622, 1.0674284, 1.0624046,
    //         1.0577925, 1.0579052, 1.0531179, 1.049944, 1.0465934, 1.0498455, 1.0436214, 1.030247, 1.0261106, 1.0270528,
    //         1.0245769, 1.0192511, 1.0119505, 1.0105906, 1.0068846, 1.0072116, 1.005775, 1.0015509, 0.99870896,
    //         1.0012114, 0.99432224, 0.9920154, 0.9861811, 0.98480856, 0.98265827, 0.979522, 0.9793139, 0.97557664,
    //         0.96723044, 0.9710303, 0.9694418, 0.96264184, 0.9592192, 0.95818007, 0.95951027, 0.9569484, 0.9504051,
    //         0.9451435, 0.9523864, 0.9543222, 0.94825166, 0.9371871, 0.9413891, 0.94003695, 0.93205225, 0.9331102,
    //         0.9320257, 0.9218203, 0.92377657, 0.9244698, 0.92655754, 0.9255434, 0.918033, 0.9200174, 0.9209533,
    //         0.91174483, 0.9124725, 0.9135132, 0.9132582, 0.9047555, 0.877947
    //     );
    //
    //     auto spectrum3 = Array<f32>::from_values(
    //         0.7302667, 0.7167722, 0.69856775, 0.70313233, 0.70892954, 0.7249322, 0.7223244, 0.685012, 0.6503742,
    //         0.6353966, 0.61641395, 0.5843986, 0.5643416, 0.54850173, 0.52982086, 0.51803625, 0.50165874, 0.4865659,
    //         0.464916, 0.45910236, 0.4451406, 0.43375838, 0.43028006, 0.42610496, 0.427322, 0.4211362, 0.4167435,
    //         0.41500434, 0.42388758, 0.43544903, 0.4390819, 0.4462397, 0.45609125, 0.46380138, 0.46871346, 0.47380018,
    //         0.4839471, 0.4919178, 0.4908653, 0.48330152, 0.48031515, 0.488765, 0.47980678, 0.47245696, 0.46473038,
    //         0.45303446, 0.43908924, 0.42949733, 0.42625123, 0.42010295, 0.42093596, 0.42323288, 0.42265865, 0.42570996,
    //         0.43360564, 0.4468926, 0.45626044, 0.46442318, 0.4691853, 0.47757566, 0.4759709, 0.47890884, 0.475628,
    //         0.469348, 0.4589533, 0.44943, 0.44269648, 0.4375622, 0.4230994, 0.42226547, 0.4272525, 0.43345204,
    //         0.43671274, 0.44831076, 0.4522518, 0.45742813, 0.46123376, 0.4634457, 0.4618885, 0.4556478, 0.45155135,
    //         0.44958544, 0.44024774, 0.43099964, 0.4295705, 0.43044007, 0.4319616, 0.42691144, 0.43515554, 0.43971556,
    //         0.44621682, 0.4544728, 0.45578498, 0.4533381, 0.45074302, 0.44701785, 0.4433896, 0.43508422, 0.4328824,
    //         0.43489406, 0.4351903, 0.43703735, 0.43663135, 0.4436209, 0.4469694, 0.45315886, 0.45435396, 0.45311952,
    //         0.44967598, 0.4430284, 0.4414776, 0.4389429, 0.43761432, 0.43912122, 0.43820685, 0.43873718, 0.4399707,
    //         0.44287932, 0.44864088, 0.4522701, 0.4501176, 0.44754177, 0.44515616, 0.4400612, 0.43895113, 0.43935436,
    //         0.4400475, 0.44174275, 0.44922972, 0.45305645, 0.45013592, 0.44975826, 0.45098263, 0.44462124, 0.44084024,
    //         0.44040594, 0.4413898, 0.4471797, 0.44476753, 0.4476936, 0.44802368, 0.4513694, 0.4509415, 0.4463156,
    //         0.44671887, 0.44810137, 0.4448169, 0.44666034, 0.450124, 0.44845253, 0.45190433, 0.45482576, 0.4545021,
    //         0.45466393, 0.4501807, 0.44700685, 0.4449166, 0.44933307, 0.4530409, 0.45484862, 0.45505437, 0.45709985,
    //         0.45480567, 0.45483035, 0.4528141, 0.45569628, 0.45414457, 0.4526779, 0.45463374, 0.45613152, 0.45725894,
    //         0.4580353, 0.45988506, 0.45953578, 0.46130967, 0.46114963, 0.46222496, 0.4603514, 0.46126303, 0.46570238,
    //         0.4639678, 0.46552408, 0.4624691, 0.4656402, 0.46302962, 0.4596583, 0.4589405, 0.46237862, 0.4678347,
    //         0.46426404, 0.46383062, 0.46702456, 0.46664146, 0.46669725, 0.46409672, 0.46960497, 0.46886432, 0.47211495,
    //         0.47128284, 0.4700576, 0.46821326, 0.47044894, 0.46932793, 0.47205, 0.4742482, 0.47238746, 0.4734335,
    //         0.47708374, 0.4786162, 0.47823763, 0.47624248, 0.47693104, 0.4806928, 0.48162365, 0.4847088, 0.48365998,
    //         0.48314703, 0.4843485, 0.48454326, 0.48466122, 0.48674235, 0.49011, 0.4864324, 0.46258432
    //         );
    //
    //     auto baseline = ctf::Baseline{};
    //     baseline.fit(spectrum.span_1d().as_const(), fftfreq_range, {0.02, 0.5});
    //
    //     const auto path = Path("~/Tmp/quinoa/midpoints.txt");
    //     save_plot_xy(noa::Linspace{fftfreq_range[0], fftfreq_range[1]}, spectrum, path, {.label = "spectrum"});
    //     std::vector<f64> background;
    //     for (i32 i{}; i < spectrum.size(); ++i) {
    //         const auto fftfreq_step = (fftfreq_range[1] - fftfreq_range[0]) / static_cast<f64>(spectrum.size() - 1);
    //         auto fftfreq = static_cast<f64>(i) * fftfreq_step;
    //         background.push_back(baseline.sample_at(fftfreq));
    //     }
    //     save_plot_xy(noa::Linspace{fftfreq_range[0], fftfreq_range[1]}, background, path, {.label = "background"});
    //
    //     // const auto shape = Shape4{1, 1, 1, 2048};
    //     // const auto simulated_ctf = Array<f32>(shape.rfft());
    //     // const auto ctf = CTFIsotropic64::Parameters{
    //     //     .pixel_size = 2.5,
    //     //     .defocus = 3.4,
    //     //     .voltage = 300.,
    //     //     .amplitude = 0.1,
    //     //     .cs = 2.7,
    //     //     .phase_shift = 0,
    //     //     .bfactor = -300,
    //     //     .scale = 1,
    //     // }.to_ctf();
    //     // ns::ctf_isotropic<"h">(simulated_ctf, shape, ctf, {.ctf_squared = true});
    //     // baseline.fit(simulated_ctf.span_1d(), {0., 0.5}, ctf);
    //     //
    //     // const auto path = Path("~/Tmp/quinoa/midpoints.txt");
    //     // save_plot_xy(noa::Linspace{0., 0.5}, simulated_ctf, path, {.label = "spectrum"});
    //     // std::vector<f64> background;
    //     // for (i32 i{}; i < simulated_ctf.size(); ++i) {
    //     //     const auto fftfreq_step = (fftfreq_range[1] - fftfreq_range[0]) / static_cast<f64>(simulated_ctf.size() - 1);
    //     //     auto fftfreq = static_cast<f64>(i) * fftfreq_step;
    //     //     background.push_back(baseline.spline.interpolate_at(fftfreq));
    //     // }
    //     // save_plot_xy(noa::Linspace{0., 0.5}, background, path, {.label = "background"});
    // }

    auto test04() {
        const auto spacing = 1.5;
        auto ctf = CTFAnisotropic64::Parameters{
            .pixel_size = {spacing, spacing},
            .defocus = {1.5, 0.2, 0.},
            .voltage = 300.,
            .amplitude = 0.1,
            .cs = 2.7,
            .phase_shift = 0,
            .bfactor = 0,
            .scale = 1,
        }.to_ctf();
        auto ctf_iso = CTFIsotropic64(ctf);

        auto shape = Shape4{1, 1, 2048, 2048};
        auto sim = Array<f32>(shape.rfft());
        ns::ctf_anisotropic<"hc">(sim, shape, ctf, {.ctf_squared = true});
        noa::write_image(sim, "~/Tmp/quinoa/figures/02/astig02/spectrum.mrc");

        auto fftfreq_range = noa::Linspace<f64>{0., resolution_to_fftfreq(spacing, 4.), true};
        fmt::println("fftfreq={}", fftfreq_range.stop);
        ns::ctf_anisotropic<"hc">(sim, shape, ctf, {.fftfreq_range = fftfreq_range, .ctf_squared = true});
        noa::write_image(sim, "~/Tmp/quinoa/figures/02/astig02/spectrum_truncated.mrc");

        auto polar_width = shape[3] / 2 + 1;
        const auto [rho_index, fftfreq_start] = nearest_integer_fftfreq(
            polar_width, Vec{0., fftfreq_range.stop}, resolution_to_fftfreq(spacing, 30.));
        polar_width -= rho_index;

        isize target_phi_size = 2048;
        auto rho_range = noa::Linspace<f64>{fftfreq_start, fftfreq_range.stop, true};
        auto phi_range = noa::Linspace<f64>{noa::deg2rad(-90.), noa::deg2rad(90.), true};

        auto polar = Array<f64>(Shape4{1, 1, target_phi_size, polar_width});
        nx::cubic_bspline_prefilter(sim, sim);
        nx::spectrum2polar<"hc2fc">(
                sim, shape, polar, {
                    .spectrum_fftfreq = fftfreq_range,
                    .rho_range = rho_range,
                    .phi_range = phi_range,
                    .interp = nx::Interp::CUBIC_BSPLINE,
            });
        noa::write_image(polar, "~/Tmp/quinoa/figures/02/astig02/polar.mrc");

        f64 target_bin_angle = 3.75;
        ctf::test05(
            ctf, polar_width, target_bin_angle, target_phi_size, nx::Interp::CUBIC_BSPLINE, rho_range, phi_range,
            sim.view(), shape, {0., fftfreq_range.stop}
        );
    }

    void test05() {
        const auto spacing = 2;
        auto ctf = CTFAnisotropic64::Parameters{
            .pixel_size = {spacing, spacing},
            .defocus = {-0.074471, 0., 0.},
            .voltage = 300.,
            .amplitude = 0.09,
            .cs = 2.7,
            .phase_shift = 0,
            .bfactor = 0,
            .scale = 1,
        }.to_ctf();
        auto ctf_iso = CTFIsotropic64(ctf);
        auto phase = ctf_iso.phase_at(0.144181);
        auto fftfreq = ctf_iso.fftfreq_at(-1.731725);
        Logger::status("fftfreq={}", fftfreq);
    }
}

auto main(int argc, char* argv[]) -> int {
    using namespace qn;

    // {
    //     Logger::initialize();
    //     test05();
    //     return 0;
    // }

    try {
        // Initialize the logger before doing anything else.
        Logger::initialize();
        auto timer = Logger::status_scope_time<false>("Main");

        // Parse the settings.
        auto settings = Settings{};
        if (not settings.parse(argc, argv))
            return EXIT_SUCCESS;

        // Adjust global settings.
        Logger::add_logfile(settings.files.output_directory / "quinoa.log");
        Logger::set_level(settings.compute.log_level);
        Session::set_gpu_lazy_loading();
        Session::set_thread_limit(settings.compute.n_threads);

        // Create a user-async stream for the GPU and ensure that the CPU stream is synchronous.
        if (settings.compute.device.is_gpu()) {
            Device::set_current(settings.compute.device);
            Stream::set_current(Stream(settings.compute.device, Stream::ASYNC));
        }
        Stream::set_current(Stream({}, Stream::SYNC));

        // Initialize the metadata early in case the parsing fails.
        auto metadata = Metadata::load_from_settings(settings);
        const auto basename = settings.files.stack_file.stem().string();

        // Register the input stack. The application loads the input stack many times. To save computation,
        // load the stack to memory once and save it inside a static array. The StackLoader will
        // check for it next time it needs it.
        // TODO By default, register only if file.is_compressed?
        if (settings.compute.register_stack)
            StackLoader::register_input_stack(settings.files.stack_file);

        if (settings.preprocessing.run) {
            auto scope_timer = Logger::status_scope_time("Preprocessing");

            if (not settings.preprocessing.exclude_stack_indices.empty()) {
                metadata.stack.exclude_if([&](const auto& image) {
                    for (isize e: settings.preprocessing.exclude_stack_indices)
                        if (e == image.index) {
                            Logger::info("Excluding view: index={} (tilt={:+.2f})", image.index, image.angles[1]);
                            return true;
                        }
                    return false;
                });
            }

            // TODO Hot pixels correction
            // TODO Frame alignment

            if (settings.preprocessing.exclude_blank_views) {
                detect_and_exclude_blank_views(
                    settings.files.stack_file, metadata.stack, {
                        .compute_device = settings.compute.device,
                        .output_directory = settings.files.output_directory / "diagnostics" / "preprocessing",
                    });
            }
        }

        // Alignment.
        if (settings.alignment.coarse_run or settings.alignment.ctf_run or settings.alignment.refine_run) {
            auto scope_timer = Logger::status_scope_time("Alignment");

            if (settings.alignment.coarse_run) {
                coarse_alignment(
                    settings.files.stack_file, metadata, {
                        .device = settings.compute.device,
                        .check_rotation = settings.alignment.coarse_check_rotation,
                        .fit_rotation_offset = settings.alignment.coarse_fit_rotation,
                        .fit_tilt_offset = settings.alignment.coarse_fit_tilt,
                        .fit_pitch_offset = settings.alignment.coarse_fit_pitch,
                        .output_directory = settings.files.output_directory / "diagnostics" / "coarse",
                    }
                );
            }

            if (settings.alignment.ctf_run) {
                ctf::fit(
                    settings.files.stack_file, metadata, {
                        .compute_device = settings.compute.device,
                        .output_directory = settings.files.output_directory / "diagnostics" / "ctf",

                        .patch_size_ang = 800,
                        .n_images_in_initial_average = 3,
                        .resolution_range = {30, 4.},
                        .fit_phase_shift = settings.alignment.ctf_fit_phase_shift,
                        .fit_astigmatism = settings.alignment.ctf_fit_astigmatism,
                        .fit_thickness = settings.alignment.ctf_fit_thickness,
                        .check_defocus_gradient = settings.alignment.ctf_check_defocus_gradient,

                        .fit_rotation = settings.alignment.ctf_fit_rotation,
                        .fit_tilt = settings.alignment.ctf_fit_tilt,
                        .fit_pitch = settings.alignment.ctf_fit_pitch,
                    }
                );
            }

            if (settings.alignment.refine_run) {
                refine_alignment(
                    settings.files.stack_file, metadata, {
                        .compute_device = settings.compute.device,
                        .correct_ctf = settings.alignment.refine_correct_ctf,
                        .phase_flip_strength = settings.alignment.refine_phase_flip_strength,
                        .fit_thickness = settings.alignment.refine_fit_thickness,
                        .fit_rotation_offset = settings.alignment.refine_fit_rotation,
                        .fit_tilt_offset = settings.alignment.refine_fit_tilt,
                        .fit_pitch_offset = settings.alignment.refine_fit_pitch,
                        .output_directory = settings.files.output_directory / "diagnostics" / "refine",
                    }
                );
            }

            // Save the metadata.
            const auto star_filename = settings.files.output_directory / fmt::format("{}.star", basename);
            metadata.save_star(star_filename);
            Logger::info("{} saved", star_filename);
        }

        // Postprocessing.
        if (settings.postprocessing.run) {
            auto scope_timer = Logger::status_scope_time("Postprocessing");
            post_processing(settings.files.stack_file, metadata, {
                .compute_device = settings.compute.device,
                .target_resolution = settings.postprocessing.resolution,
                .min_size = 512,
                .output_directory = settings.files.output_directory,
                .save_aligned_stack = settings.postprocessing.stack_run,
                .stack_dtype = settings.postprocessing.stack_dtype,
                .stack_correct_rotation = settings.postprocessing.stack_correct_rotation,
                .stack_interp = settings.postprocessing.stack_interpolation,

                .save_tomogram = settings.postprocessing.tomogram_run,
                .tomogram_dtype = settings.postprocessing.tomogram_dtype,
            }, {
                .ramp_filter = settings.postprocessing.tomogram_ramp_filter,
                .correct_ctf = settings.postprocessing.tomogram_correct_ctf,
                .phase_flip_strength = settings.postprocessing.tomogram_phase_flip_strength,
                .defocus_step_nm = 15,
            }, {
                .algorithm = settings.postprocessing.tomogram_algorithm,
                .z_padding_percent = settings.postprocessing.tomogram_z_padding_percent / 100,
                .correct_rotation = settings.postprocessing.tomogram_correct_rotation,
                .oversampling_factor = settings.postprocessing.tomogram_oversampling_factor,
                .interp = settings.postprocessing.tomogram_interpolation,
            });
        }
    } catch (...) {
        for (i32 i{}; auto& message : noa::Exception::backtrace())
            Logger::error("[{}]: {}", i++, message);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
