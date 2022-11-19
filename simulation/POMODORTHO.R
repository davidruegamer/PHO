#' Orthogonalize a Semi-Structured Model Post-hoc
#' 
#' @param mod deepregression model
#' @param name_penult character name of the penultimate layer 
#' of the deep part part
#' @param param_nr integer; number of the parameter to be returned
#' 
#' @return a \code{deepregression} object with weights frozen and
#' deep part specified by \code{name_penult} orthogonalized
#' 
orthog_post_fitting <- function(mod, name_penult, struct_inp, param_nr = 1)
{
  
  mod_new_keras <- tf$keras$models$clone_model(mod$model)
  
  # check if model is distributional with concat before
  ll <- mod$model$layers[[length(mod$model$layers)]]
  if(grepl("distribution_lambda", ll$name) & 
     grepl("concatenate", previous_layers(ll)$name)
  ){
    
    concat_ll <- previous_layers(ll)
    dist_param_outputs <- previous_layers(concat_ll)
    
    pll <- dist_param_outputs[[param_nr]]
    
    if(grepl("^add\\_", pll$name)){
      ppll <- previous_layers(pll)
      deep <- which(sapply(ppll, "[[", "name")==name_penult)
      warning("Function is currently only returning the unstructured layer.")
      return(ppll[[deep]])
    }else{
      stop("Model has no sum in the last layer.")
    }
    
  }else{
    
    stop("Not implemented for last layer '", gsub("\\_[0-9]+", "", ll$name),
         "' with previous layer '", 
         gsub("\\_[0-9]+", "", ll$`_inbound_nodes`[[1]]$inbound_layers$name),
         "'.")
    
  }
}

previous_layers <- function(layer)
{
  
  inbn <- layer$`_inbound_nodes`
  if(length(inbn)>1)
    stop("previous_layer function does not work on layers with multiple inbound nodes.")
  return(inbn[[1]]$inbound_layers)
  
}
