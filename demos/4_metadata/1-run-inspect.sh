
echo " "
echo " "
echo "Let's look at the objects within the PDF:"
echo " "
qpdf --show-pages pdf-puzzle.pdf
echo " "
echo " "

echo " "
echo " "
echo "Let's look at the objects within the PDF:"
echo " "
qpdf --show-object=5,0 pdf-puzzle.pdf
echo " "
echo " "

echo " "
echo " "
echo "Let's look at the objects within the PDF:"
echo " "
qpdf --show-object=5,0 --raw-stream-data pdf-puzzle.pdf
echo " "
echo "Use: https://cryptii.com/pipes/ascii85-encoding/"
echo " "
echo " "

echo " "
echo " "
echo "Find reference to the earlier version (/Prev 718 points to the previous cross-reference table at byte offset 718.):"
echo " "
qpdf --show-object=trailer pdf-puzzle.pdf
echo " "
echo " "
