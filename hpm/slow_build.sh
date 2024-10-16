CPPCOMPILER="g++-10"
if ! compiler_loc="$(type -p "${CPPCOMPILER}")" || [[ -z $compiler_loc ]]; then
	echo "Did not find g++-10. Trying simple g++ instead."
	CPPCOMPILER="g++"
fi

if (($# < 1)); then
	${CPPCOMPILER} -std=c++20 -Wno-psabi main.c++ command-line.c++ ellipse.c++ ellipse-detector.c++ find.c++ hpm.c++ marks.c++ solve-pnp.c++ util.c++ ed/ED.c++ ed/EDCircles.c++ ed/EDColor.c++ ed/EDCommon.c++ ed/EDLines.c++ ed/EDPF.c++ ed/NFA.c++ -I.. -I../extern/cppcore -I../extern/eigen -I/usr/include/opencv4 -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_calib3d -lopencv_features2d -o hpm &
	${CPPCOMPILER} -std=c++20 -Wno-psabi find.test.c++ ellipse.c++ ellipse-detector.c++ find.c++ marks.c++ util.c++ solve-pnp.c++ ed/ED.c++ ed/EDCircles.c++ ed/EDColor.c++ ed/EDCommon.c++ ed/EDLines.c++ ed/EDPF.c++ ed/NFA.c++ -I.. -I../extern/cppcore -I../extern/boostut -I../extern/eigen -I/usr/include/opencv4 -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_calib3d -o find.test &
	${CPPCOMPILER} -std=c++20 -Wno-psabi hpm.test.c++ hpm.c++ ellipse.c++ ellipse-detector.c++ find.c++ marks.c++ util.c++ solve-pnp.c++ ed/ED.c++ ed/EDCircles.c++ ed/EDColor.c++ ed/EDCommon.c++ ed/EDLines.c++ ed/EDPF.c++ ed/NFA.c++ -I.. -I../extern/cppcore -I../extern/boostut -I../extern/eigen -I/usr/include/opencv4 -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_calib3d -o hpm.test &
	${CPPCOMPILER} -std=c++20 -Wno-psabi marks.test.c++ ellipse.c++ marks.c++ util.c++ solve-pnp.c++ ellipse-detector.c++ ed/ED.c++ ed/EDCircles.c++ ed/EDColor.c++ ed/EDCommon.c++ ed/EDLines.c++ ed/EDPF.c++ ed/NFA.c++ -I.. -I../extern/cppcore -I../extern/boostut -I../extern/eigen -I/usr/include/opencv4 -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_calib3d -o marks.test &

	wait $(jobs -pr)
	echo "Ca 50% finished..."

	${CPPCOMPILER} -std=c++20 -Wno-psabi solve-pnp.test.c++ ellipse.c++ marks.c++ util.c++ solve-pnp.c++ ed/ED.c++ ed/EDCircles.c++ ed/EDColor.c++ ed/EDCommon.c++ ed/EDLines.c++ ed/EDPF.c++ ed/NFA.c++ -I.. -I../extern/cppcore -I../extern/boostut -I../extern/eigen -I/usr/include/opencv4 -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_calib3d -o solve-pnp.test &
	${CPPCOMPILER} -std=c++20 -Wno-psabi util.test.c++ ellipse.c++ marks.c++ solve-pnp.c++ util.c++ ed/ED.c++ ed/EDCircles.c++ ed/EDColor.c++ ed/EDCommon.c++ ed/EDLines.c++ ed/EDPF.c++ ed/NFA.c++ -I.. -I../extern/cppcore -I../extern/boostut -I../extern/eigen -I/usr/include/opencv4 -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_calib3d -o util.test &
	${CPPCOMPILER} -std=c++20 -Wno-psabi ed/ED.test.c++ ed/ED.c++ ed/EDCircles.c++ ed/EDColor.c++ ed/EDCommon.c++ ed/EDLines.c++ ed/EDPF.c++ ed/NFA.c++ -I.. -I../extern/cppcore -I../extern/boostut -I../extern/eigen -I/usr/include/opencv4 -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -o ed/ED.test &

	wait $(jobs -pr)
else
	${CPPCOMPILER} -std=c++20 -Wno-psabi main.c++ command-line.c++ ellipse.c++ ellipse-detector.c++ find.c++ hpm.c++ marks.c++ solve-pnp.c++ util.c++ ed/ED.c++ ed/EDCircles.c++ ed/EDColor.c++ ed/EDCommon.c++ ed/EDLines.c++ ed/EDPF.c++ ed/NFA.c++ -I.. -I../extern/cppcore -I../extern/eigen -I/usr/include/opencv4 -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_calib3d -lopencv_features2d -o hpm
fi

echo "Finished"
