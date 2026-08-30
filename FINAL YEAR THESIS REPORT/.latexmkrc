# Run makeindex for the nomencl package (.nlo -> .nls) during latexmk builds
add_cus_dep('nlo', 'nls', 0, 'nlo2nls');
sub nlo2nls {
    system("makeindex -s nomencl.ist -o \"$_[0].nls\" \"$_[0].nlo\"");
}
push @generated_exts, 'nlo', 'nls';
