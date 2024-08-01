using Pkg

# # Install all required packages in one go
Pkg.add([
    "DataFrames",
    "LinearAlgebra",
    "GLM",
    "DataFramesMeta",
    "StatsModels",
    "CategoricalArrays",
    "Distributions",
    "Random",
    "FixedEffectModels",
    "StatsBase",
    "Statistics",
    "StatFiles",
    "StatsFuns"
])

# Precompile installed packages
Pkg.precompile()

using DataFrames
using LinearAlgebra
using GLM
using DataFramesMeta
using StatsModels
using CategoricalArrays
using Distributions
using Random
using FixedEffectModels
using StatsBase
using Statistics
using StatFiles
using StatsFuns

# Defining custom vech function equivalent to the one in R
function vech(A::AbstractMatrix)
    # finds the nrows of the matrix A
    n = size(A, 1)
    # defines v to be a vector whose elements are the same type 
    # as those found in A, and give it the size of the lower
    # triangular
    v = Vector{eltype(A)}(undef, (n * (n + 1)) ÷ 2)
    # setting the start of the v vector's indices
    k = 1
    
    # looping over 
    @inbounds for j in 1:n, i in j:n
        v[k] = A[i, j]
        k += 1
    end
    return v
end

# Function to extract the column name from factor(column_name)
function extract_factor_column(s::AbstractString)
    # defining an expression to search for
    regex = r"factor\s*\(\s*([^)\s]+)\s*\)"
    # checking for a match
    m = match(regex, s)
    # returning nothing or the column_name
    return m === nothing ? nothing : m.captures[1]
end

# # For use with XVARSX, XVARAX, XVARBX to replace "function()" with "term()" so that glm understands it
function replace_factor(s::String)
    # defining the pattern to look for
    pattern = r"factor\s*\(\s*([^)]+)\s*\)"
    # replacing the pattern `factor(...)` 
    return replace(s, pattern => s"\1")
end

function VECHTEST(df::DataFrame, y::String, 
        xv::String, cv::String, 
        fc::String, cc::String)
    
    # defining temporary pointer nodes
    stard=df
    CTEMP=cv
    XTEMP=xv
    y=y
    null=fc
    alt1=cc
    
    # making the fixed effect x-variables into an equation-type string 
    rightC = join(split(CTEMP, r"\s+"), "+")
    #making the X-variables of interest into an equation-type string
    rightX = join(split(XTEMP, r"\s+"), "+")
    # putting the two strings together
    right = join([rightX, rightC], "+")
    
    # For use in XVARSX... replacing "factor(variable)"" with just the variable name
    # in the strings and converting the data values to type = string so that feols can
    # be used without issue.
    # extracting just the column name without the `factor(`column_name`)` part
    factor_column = extract_factor_column(right)
    
    if factor_column != nothing
        stard[!, factor_column] = string.(stard[!, factor_column])
        right = replace_factor(right)
        rightC = replace_factor(rightC)
    end
    
    # Converting the column to a categorical array and then to 
    # integer indices, storing in new column `calt`
    stard.calt = levelcode.(CategoricalArray(stard[:, alt1]))

    # Calculate the maximum value of `calt`
    G = maximum(stard.calt)
    
    # Converting the column `null` to a categorical array and then to 
    # integer indices, storing in new column `cnul`
    stard.cnul = convert(Vector{Int}, CategoricalArray(stard[!, null]))
    
    # new temporary dataframe node
    temp = stard
    
    # Group by 'calt' and 'cnul'
    # create 'cros' column with unique group IDs
    temp.cros = groupindices(groupby(temp, [:calt, :cnul]))
    
    # Add 'cros' column from 'temp' to 'stard'
    stard.cros = temp.cros
    
    # Calculate the maximum value of `cros`
    H = maximum(stard.cros)
    
    # sorting stard by `calt` and `cros`
    sort!(stard, [:calt, :cros])
    
    # making a new dataframe that includes a groupID on the stard
    # dataframe
    groups = groupby(stard, [:calt, :cros])
    dt = combine(groups) do sdf
        sdf.id = 1:nrow(sdf)
        return sdf
    end

    # defining column `ccnt` that is that new dataframe's groupID column
    stard.ccnt = dt.id
    
    # defining linear regression formula as a string
    frm = y * " ~ " * right
    # converting string-formula to a formula type for use in lm()
    flm = @eval(@formula($(Meta.parse(frm))))
    
    # Fitting a linear model
    model = lm(flm, stard)
    
    # extracting values from the regression
    stard.tempres = residuals(model)
    stard.tempfit = fitted(model)
    NMK = dof_residual(model)
    N = nobs(model)
    K = N-NMK
    
    # defining variables of interest
    xterms = split(rightX, '+')
    # finding how many variables of interest there are
    xnum = length(xterms)
    
    # making a vector of placeholders numbered for each var of interest
    cns = ["tempsc$i" for i in 1:xnum]
    
    # defining new dataframe with columns: `calt`,`cnul`, and `cros`
    # from the stard dataframe
    VC = DataFrame(
        calt = stard.calt,
        cnul = stard.cnul,
        cros = stard.cros
    )
    
    # defining new temporary dataframe named orid
    orid = stard
    
    # fitting an lm without the vars of interest
    for x in xterms
        flmtemp = @eval(@formula($(Meta.parse("$x ~ $rightC"))))
        mdltemp = lm(flmtemp, orid)
        VC[!, Symbol("tempres_$x")] = stard.tempres .* residuals(mdltemp)
    end
    
    # setting values to rename the VC dataframe's columns and then renaming them
    l = 4
    r = ncol(VC)
    rename(VC, Dict(zip(names(VC)[l:r], cns)))
    
    
    M_A = (G/(G - 1)) * ((N-1)/NMK)
    M_F = (N-1)/(NMK)*H/(H-1)
    Gk = zeros(xnum^2, div(xnum * (xnum + 1), 2))
        
    
    for i in 1:xnum
        for j in i:xnum
            a = (j - 1) * xnum + i
            b = (i - 1) * xnum + j
            c = (j - 1) * (xnum - j ÷ 2) + i
            Gk[a, c] = 1
            Gk[b, c] = 1
        end
    end
    
    # Hk calculation
    Hk = inv(transpose(Gk) * Gk) * transpose(Gk)
    
    # Initialize matrices
    temp_sumg = zeros(xnum, xnum)
    temp_num_alt = zeros(xnum, xnum)
    
    var_right = zeros(size(Hk, 1), size(Hk, 1))
    var_left = zeros(size(Hk, 1), size(Hk, 1))
            
    ALT = zeros(xnum, xnum)
    NLL = zeros(xnum, xnum)
    
    theta = zeros(size(Hk, 1), 1)
    
    #aggregation
    cols_to_aggregate = names(VC)[l:r]
    
    sh_h = combine(groupby(VC, Symbol("cros")),
               [Symbol(col) => sum => Symbol(col) for col in cols_to_aggregate])
    sg = combine(groupby(VC, Symbol("calt")),
               [Symbol(col) => sum => Symbol(col) for col in cols_to_aggregate])

    # Extract specific columns
    lx = 2
    lr = lx + xnum - 1
    sh_h = Matrix(sh_h[:, lx:lr])
    sg = Matrix(sg[:, lx:lr])
    
    for g in 1:G
        temp_sumh = zeros(xnum, xnum)
        temp_var_left =zeros(xnum, xnum)
        temp_var_right =zeros(xnum, xnum)
    
        temp_sg = sg[g,:]
        temp_num_alt = temp_sg * transpose(temp_sg)
        ALT = ALT + temp_num_alt
    
        idx = Vector(stard[findall((stard.calt .== g) .& (stard.ccnt .== 1)), :cros])
    
        for i in idx
            sh1 = sh_h[i, :]
            temp_cross = sh1 * sh1'
    
            # var left
            temp_sumh = temp_sumh + temp_cross
    
            # var right
            temp_var_right = Hk * kron(temp_cross, temp_cross) * Hk'
            var_right = temp_var_right + var_right
        end

        # var left
        temp_var_left = Hk * kron(temp_sumh, temp_sumh) * Hk'
        var_left = temp_var_left .+ var_left

        temp_sumg = temp_sumg .+ temp_sumh
    end
    
    NLL= temp_sumg
    NLL = M_F*NLL
    ALT = M_A*ALT

    theta = vech(ALT - NLL)

    var_left = 2 * var_left
    var_right = 2 * var_right
    var = var_left - var_right
    
    if xnum == 1
        tau = first(theta) / sqrt(first(var))
    else
        tau = dot(theta, inv(var), theta)
    end

    chi_df = xnum * (xnum + 1) / 2
    
    return (
        Dict(
            :H => H,
            :G => G,
            :xn => xnum,
            :XV => XTEMP,
            :CV => CTEMP,
            :theta => theta,
            :tau => tau,
            :var => var,
            :chi_df => chi_df,
            :data => stard
        )
    )
end

function MNWTEST(df::DataFrame, y::String, 
        xv::String, cv::String, 
        fc::String, cc::String, b::Int)
    


    # Defining temporary pointers
    df = df
    y = y
    XTEMP = xv
    CTEMP = cv
    null = fc
    alt1 = cc
    B = b
    
    # calling the VECHTEST function
    ls = VECHTEST(df,y,XTEMP, CTEMP, null, alt1)
    
    # extracting VECHTEST values
    xnum = getindex(ls,:xn)
    tauhat = getindex(ls, :tau)
    chi_df = getindex(ls, :chi_df)
    
    if xnum == 1
        MNW_P = 2 * min(normcdf(tauhat), 1-normcdf(tauhat))
    else
        if xnum >=2
            MNW_P = 1-chisqcdf(chi_df, tauhat)
        end
    end
    
    dt = getindex(ls, :data)
    
    # Sorting dt by `null`
    sort(dt,null)
    
    # extracting some values
    temper = dt[:,:tempres]
    tempft = dt[:,:tempfit]
    
    taus = fill(1.0, B)
    
    Random.seed!(42)
    for i in 1:B
        
        # new temporary dataframe node
        dg = dt
            
        
        # Group by `null`, then add a new column `uni` with 
        # random uniform values
        dg = transform(groupby(dt, null)) do sdf
            (uni = rand(nrow(sdf)),)
        end
        
        temp_uni = dg.uni
        
        temp_pos = temp_uni .<0.5
        temp_ernew = (2 .*temp_pos .-1).*temper
        temp_ywild = tempft + temp_ernew
        
        dt.booty=temp_ywild
    
        lstemp= VECHTEST(dt,"booty",XTEMP,CTEMP,null,alt1)
  
        taus[i] = getindex(lstemp,:tau)
    end
    
    if xnum == 1
    
        temp_rej = abs(tauhat) .<= abs.(taus)
    else
        temp_rej =  tauhat .<= taus
    end

    temp_U = fill(1, length(temp_rej), 1)
    temp_sum = transpose(temp_U)*temp_rej
    boot_p = temp_sum / length(temp_rej)
    
    return (
        Dict(
            :H => getindex(ls,:H),
            :G => getindex(ls,:G),
            :theta => getindex(ls, :theta),
            :tau => getindex(ls, :tau),
            :chi_df => getindex(ls,:chi_df),
            :MNW_P => MNW_P,
            :bp => boot_p
            )
        )
end


function lcat(lsmnw::Dict)
    return Dict(
        "H" => getindex(lsmnw, :H),
        "G" => getindex(lsmnw, :G),
        "theta" => join(string.(getindex(lsmnw, :theta)), " "),
        "tau" => getindex(lsmnw, :tau),
        "chi_df" => getindex(lsmnw, :chi_df),
        "MNW_P" => getindex(lsmnw, :MNW_P),
        "bp" => getindex(lsmnw, :bp)
    )
end

function IMTEST(df::DataFrame, y::AbstractString, 
        xv::AbstractString, cv::AbstractString,fr::AbstractString,
        fc::AbstractString, cc::AbstractString, tm::Int)

    df=df
    y=y 
    xv=xv # xs or xa
    cv=cv # XVARSI or XVARAI
    fr= fr # "schid1n" # schid1n
    fc= fc # "newid" # null 'newid' or 'clsid'
    cc= cc #"schid1n" # alt1 `schid1n`

    
    # Specify the variable of interest
    variable_of_interest = Symbol(xv)  # Replace with your variable


    # Convert the `alternative's` column name to a Symbol
    alt = Symbol(cc)
    # Convert the `null's` column name to a Symbol
    null = Symbol(fc)
    # Group the DataFrame by the `alternative`
    grouped_df = groupby(df, alt)

    # Create a dictionary to map group values to group numbers
    group_values = unique(df[!,alt])
    group_numbers = Dict(value => idx for (idx, value) in enumerate(group_values))
    
    # Create a new column for group numbers and assign values
    df.group_number = [group_numbers[row[alt]] for row in eachrow(df)]


    
    j = maximum(df.group_number);
    
    beta = zeros(j,1);
    omega = zeros(j,1);
    
    # Split the fixed_effects string by spaces to get individual elements
    cvs = split(cv)
    # Mapping each element to the format "fe(element)"
    #formatted_Cs = ["fe($e)" for e in cvs]
    formatted_Cs = ["$e" for e in cvs]
    # making the fixed effect x-variables into an equation-type string 
    rightC = join(formatted_Cs, "+")
    #making the X-variables of interest into an equation-type string
    rightX = join(split(xv, r"\s+"), "+")
    # putting the two strings together to form the RHS of the regression EQN
    right = join([rightX, rightC], "+")   
    right2 = "$right + fe($fr)"

    # joining the LHS and RHS to make the full EQN
    frm = join([y, right2], "~")
    # making the formula readable in reg()
    flm = @eval(@formula($(Meta.parse(frm))))

    for g in 1:j
        
        temp_g = df[df.group_number .== g, :]
        # Create the regression model
        fs = reg(temp_g, flm, Vcov.cluster(null), save = true)
    
        # Extract coefficients
        coefs = coef(fs)
        # Extract standard errors
        se = stderror(fs)
    
        # extracting the coefficients' names
        vec = coefnames(fs)
        
        # making the variable of interest searchable in the coefficients' names
        search_str = "$xv"

        # Creating a regular expression pattern that searches for the variable of interest
        pattern = Regex("$(search_str).*")

        # Finding the index of the first occurrence that matches the pattern
        index = findfirst(x -> occursin(pattern, x), vec)

        # Get the coefficient and standard error for the specific variable given the index
        coefficient = coefs[index]
        standard_error = se[index]
        
        # setting the coef and se to the g-th spot in the beta and omega vectors'
        beta[g, 1] = coefficient
        omega[g, 1] = standard_error
    
    end

    for i in 1:length(beta)
        if beta[i] == NaN
            beta[i] = 0
        end
    end
    for i in 1:length(omega)
        if omega[i] == NaN
            omega[i] = 0
        end
    end
    
    beta[isnan.(beta)] .= 0
    omega[isnan.(omega)] .= 0

    s2 = var(beta)
    
    time = tm
    
    ybar = fill(NaN, time, 1)
    
    for k in 1:size(ybar,1)
        yj = omega .* quantile.(Normal(), rand(j))
        avey = mean(yj)
        sy2 = (yj .- avey) .* (yj .- avey)
        
        ybk = (1/(j - 1))*sum(sy2)
        
        ybar[k, 1] = ybk
    end
    
    temp_rej = s2 .< ybar
    temp_U = ones(size(temp_rej, 1), 1)
    temp_sum = transpose(temp_U) * temp_rej
    IM_p = temp_sum / size(temp_rej, 1)
    
    return (
    Dict(
        :S2 => s2,
        :IM_p => IM_p
        )
    )
end

# Specify the full file path
file_path = raw"C:\Users\ryan-\OneDrive\Documents\vCarleton Summer RA Work\Julia Project\Tonghui Rcode\star_test.dta"

# Read the Stata file into a DataFrame
df = DataFrame(load(file_path))

# Testing the VECHTEST function

B=500;

XVARS="aide_1 treadssk male nonwhite teach_nonwhite totexp1 freelunch brys2 brys3 brys5 sbq2 sbq3 sbq4 hdg2 hdg3 hdg4";
XVARA="small_1 treadssk male nonwhite teach_nonwhite totexp1 freelunch brys2 brys3 brys5 sbq2 sbq3 sbq4 hdg2 hdg3 hdg4";
XVARB="treadssk male nonwhite teach_nonwhite totexp1 freelunch brys2 brys3 brys5 sbq2 sbq3 sbq4  hdg2 hdg3 hdg4";


y="treadss1";
  
xs="small_1";
xa="aide_1"; 
  
xb="small_1 aide_1";

fc = "newid";

cc = "clsid";

XVARSX="aide_1 treadssk male nonwhite teach_nonwhite totexp1 freelunch brys2 brys3 brys5 sbq2 sbq3 sbq4 hdg2 hdg3 hdg4 factor(schid1n)";
XVARAX="small_1 treadssk male nonwhite teach_nonwhite totexp1 freelunch brys2 brys3 brys5 sbq2 sbq3 sbq4 hdg2 hdg3 hdg4 factor(schid1n)";
XVARBX="treadssk male nonwhite teach_nonwhite totexp1 freelunch brys2 brys3 brys5 sbq2 sbq3 sbq4  hdg2 hdg3 hdg4 factor(schid1n)";

VECHTEST_time = @elapsed begin
v1 = VECHTEST(df, y, xs, XVARS, "clsid", "schid1n");
end;
println("Elapsed time: $VECHTEST_time seconds")

println(v1)

## MNWTEST TESTING

B=500;

XVARS="aide_1 treadssk male nonwhite teach_nonwhite totexp1 freelunch brys2 brys3 brys5 sbq2 sbq3 sbq4 hdg2 hdg3 hdg4";
XVARA="small_1 treadssk male nonwhite teach_nonwhite totexp1 freelunch brys2 brys3 brys5 sbq2 sbq3 sbq4 hdg2 hdg3 hdg4";
XVARB="treadssk male nonwhite teach_nonwhite totexp1 freelunch brys2 brys3 brys5 sbq2 sbq3 sbq4  hdg2 hdg3 hdg4";


y="treadss1"; 
  
xs="small_1";
xa="aide_1";
  
xb="small_1 aide_1";

fc = "newid";

cc = "clsid";

# ls1= MNWTEST(df,y,xs,XVARS,"newid","clsid",B)
# ls2= MNWTEST(df,y,xs,XVARS,"newid","schid1n",B)
# ls3= MNWTEST(df,y,xs,XVARS,"clsid","schid1n",B)

elapsed_time = @elapsed begin

ls4= MNWTEST(df,y,xa,XVARA,"newid","clsid",B);
    
end;

elapsed_time2 = @elapsed begin
# ls5= MNWTEST(df,y,xa,XVARA,"newid","schid1n",B)
ls6= MNWTEST(df,y,xa,XVARA,"clsid","schid1n",B);

# ls7=MNWTEST(df,y,xb,XVARB,"newid","clsid",B)
# ls8=MNWTEST(df,y,xb,XVARB,"newid","schid1n",B)
# ls9=MNWTEST(df,y,xb,XVARB,"clsid","schid1n",B)
# 
# XVARSX="aide_1 treadssk male nonwhite teach_nonwhite totexp1 freelunch brys2 brys3 brys5 sbq2 sbq3 sbq4 hdg2 hdg3 hdg4 factor(schid1n)"
# XVARAX="small_1 treadssk male nonwhite teach_nonwhite totexp1 freelunch brys2 brys3 brys5 sbq2 sbq3 sbq4 hdg2 hdg3 hdg4 factor(schid1n)"
# XVARBX="treadssk male nonwhite teach_nonwhite totexp1 freelunch brys2 brys3 brys5 sbq2 sbq3 sbq4  hdg2 hdg3 hdg4 factor(schid1n)"

# ls10= MNWTEST(df,y,xs,XVARSX,"newid","clsid",B)
# ls11= MNWTEST(df,y,xs,XVARSX,"newid","schid1n",B)
# ls12= MNWTEST(df,y,xs,XVARSX,"clsid","schid1n",B)

# ls13= MNWTEST(df,y,xa,XVARAX,"newid","clsid",B)
# ls14= MNWTEST(df,y,xa,XVARAX,"newid","schid1n",B)
# ls15= MNWTEST(df,y,xa,XVARAX,"clsid","schid1n",B)

# ls16= MNWTEST(df,y,xb,XVARBX,"newid","clsid",B)
# ls17= MNWTEST(df,y,xb,XVARBX,"newid","schid1n",B)
# ls18= MNWTEST(df,y,xb,XVARBX,"clsid","schid1n",B)

    
end;


# Print the result
println("Elapsed time: $elapsed_time seconds")
println("Elapsed time: $elapsed_time2 seconds")
println(getindex(ls4, :tau))
println(getindex(ls6, :tau))

lcat(ls4)

## IMTEST TESTING
df = DataFrame(load(file_path));

y="treadss1";

xs="small_1";
xa="aide_1";
fr="schid1n";

nulla="newid";
nullb="clsid";
alt1="schid1n";

XVARSI="aide_1 treadssk male nonwhite teach_nonwhite totexp1 freelunch brys2 brys3 brys5 sbq2 sbq3 sbq4 hdg2 hdg3 hdg4";
XVARAI="small_1 treadssk male nonwhite teach_nonwhite totexp1 freelunch brys2 brys3 brys5 sbq2 sbq3 sbq4 hdg2 hdg3 hdg4";
fatr="schid1n";

tm = 99;
 

elapsed_time3 = @elapsed begin
    
im1 = IMTEST(df, y, xs, XVARSI, fr, nulla, alt1, 99);

df = DataFrame(load(file_path));
im2 = IMTEST(df, y, xs, XVARSI, fr, "clsid", alt1, 99);

df = DataFrame(load(file_path));
im3 = IMTEST(df, y, xa, XVARAI, fr, nulla, alt1, 99);

df = DataFrame(load(file_path));
im4 = IMTEST(df, y, xa, XVARAI, fr, "clsid", alt1, 99);
end;

println("Elapsed time: $elapsed_time3 seconds")
println("")
print(im1)
println("")
print(im2)
println("")
print(im3)
println("")
print(im4)


